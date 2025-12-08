# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import inspect
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional

import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.agent_dispatcher.domain.plan.action.skill.mcp.engine import MCPEngine
from app.agent_dispatcher.infrastructure.entity.AgentInstance import AgentInstance
from app.cosight.agent.base.skill_to_tool import convert_skill_to_tool,get_mcp_tools,convert_mcp_tools
from app.cosight.llm.chat_llm import ChatLLM
from app.cosight.task.time_record_util import time_record
from app.cosight.tool.tool_result_processor import ToolResultProcessor
from app.cosight.task.plan_report_manager import plan_report_event_manager
from app.common.logger_util import logger
from app.cosight.agent.base.tool_arg_mapping import FUNCTION_ARG_MAPPING


class BaseAgent:
    def __init__(self, agent_instance: AgentInstance, llm: ChatLLM, functions: {}, plan_id: str = None,
                 draft_llm: Optional[ChatLLM] = None, verifier_llm: Optional[ChatLLM] = None,
                 use_draft_verifier: Optional[bool] = None):
        self.agent_instance = agent_instance
        self.llm = llm
        self.draft_llm = draft_llm
        self.verifier_llm = verifier_llm
        # Check environment variable if use_draft_verifier not explicitly set
        if use_draft_verifier is None:
            use_draft_verifier = os.environ.get("USE_DRAFT_VERIFIER", "false").lower() == "true"
        self.use_draft_verifier = use_draft_verifier and (draft_llm is not None)
        self.tools = []
        self.mcp_tools = []
        self.mcp_tools = get_mcp_tools(self.agent_instance.template.skills)
        for skill in self.agent_instance.template.skills:
            self.tools.extend(convert_skill_to_tool(skill.model_dump(), 'en'))
        self.tools.extend(convert_mcp_tools(self.mcp_tools))
        self.functions = functions
        self.history = []
        self.plan_id = plan_id
        self._tool_event_sequence = 0  # 工具事件序列号
        self._file_saver_call_count = {}  # 记录每个步骤的file_saver调用次数
        # Only set plan to None if it hasn't been set by subclass
        if not hasattr(self, 'plan'):
            self.plan = None  # Will be set by subclasses that have access to Plan

    def _normalize_tool_args(self, function_to_call, raw_args: Dict[str, Any], function_name: str = "") -> Dict[str, Any]:
        """
        将LLM生成的可能不规范的参数键统一映射为工具函数真实参数名。

        规则：
        - 基于目标函数签名的参数集合，仅对存在于签名的参数进行填充
        - 使用通用别名表进行匹配（如 file->filename, filepath->filename, text->content 等）
        - 支持无下划线/大小写不敏感匹配
        - 不在签名内的键保持原样（以便函数可接收 **kwargs）
        """
        try:
            signature = inspect.signature(function_to_call)
            param_names = set(signature.parameters.keys())

            def normalize_key(k: str) -> str:
                return (k or '').replace('_', '').lower()

            # 仅函数名级映射：alias(lower)->canonical
            alias_reverse = {}
            fn_key = (function_name or '').lower()
            mapping_cfg = FUNCTION_ARG_MAPPING.get(fn_key, {})
            aliases_cfg = mapping_cfg.get('aliases', {})
            for canonical, aliases in aliases_cfg.items():
                alias_reverse[normalize_key(canonical)] = canonical
                for a in aliases:
                    alias_reverse[normalize_key(a)] = canonical

            normalized_args: Dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}

            # 将别名键映射到签名中的canonical键；保留未映射键
            produced: Dict[str, Any] = {}
            used_keys = set()

            for key, val in list(normalized_args.items()):
                key_norm = normalize_key(key)

                # 如果原键就在签名里，直接使用
                if key in param_names:
                    produced[key] = val
                    used_keys.add(key)
                    continue

                # 尝试用别名反查canonical
                if key_norm in alias_reverse:
                    canonical = alias_reverse[key_norm]
                    if canonical in param_names and canonical not in produced:
                        produced[canonical] = val
                        used_keys.add(key)
                        logger.info(f"Tool args normalized: {key} -> {canonical}")
                        continue

                # 尝试模糊：将签名参数做无下划线匹配
                for p in param_names:
                    if normalize_key(p) == key_norm and p not in produced:
                        produced[p] = val
                        used_keys.add(key)
                        logger.info(f"Tool args normalized (fuzzy): {key} -> {p}")
                        break

            # 把未用上的原始键（可能用于 **kwargs）补回
            for key, val in normalized_args.items():
                if key not in used_keys and key not in produced:
                    produced[key] = val

            # 必填项校验（若配置了 required）
            required = mapping_cfg.get('required', [])
            missing = [r for r in required if r in param_names and r not in produced]
            if missing:
                logger.warning(f"Missing required args for {function_name}: {missing}")

            return produced
        except Exception as e:
            logger.warning(f"args normalization failed: {e}")
            return raw_args

    def find_mcp_tool(self, tool_name):
        for tool in self.mcp_tools:
            for func in tool['mcp_tools']:
                if func.name == tool_name:
                    return tool, func.name
        return None

    def _push_tool_event(self, event_type: str, tool_name: str, tool_args: str = "", 
                        tool_result: str = "", step_index: int = None, duration: float = None, 
                        error: str = None):
        """
        推送工具执行事件到队列
        
        Args:
            event_type: 事件类型 ('tool_start', 'tool_complete', 'tool_error')
            tool_name: 工具名称
            tool_args: 工具参数
            tool_result: 工具结果
            step_index: 步骤索引
            duration: 执行耗时（秒）
            error: 错误信息
        """
        try:
            # 放宽早退条件：只要有 plan_id 即可上报；若无 plan_id 则不发布
            if not getattr(self, 'plan_id', None):
                return
            
            # 过滤MCP工具事件：不发送step_index=-1的MCP工具事件到前端
            if step_index == -1:
                logger.debug(f"跳过MCP工具事件发送到前端: {event_type} for {tool_name}")
                return
            
            # 增加序列号确保事件顺序
            self._tool_event_sequence += 1
            
            # 构建事件数据
            event_data = {
                "event_type": event_type,
                "tool_name": tool_name,
                "tool_name_zh": self._get_tool_name_zh(tool_name),
                "tool_args": tool_args,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step_index": step_index,
                "sequence": self._tool_event_sequence  # 添加序列号
            }
            
            if duration is not None:
                event_data["duration"] = round(duration, 2)
            
            if error:
                event_data["error"] = error
            elif tool_result:
                # 处理工具结果
                task_title = self.plan.title if self.plan else ""
                processed_result = ToolResultProcessor.process_tool_result(tool_name, tool_args, tool_result, task_title)
                event_data["processed_result"] = processed_result
                event_data["raw_result_length"] = len(tool_result)
                
                # 注入验证信息，包含URL
                self._inject_verification_info(event_data, tool_name, processed_result)
            
            # 携带路由键（plan_id）发布事件
            plan_report_event_manager.publish("tool_event", self.plan_id, event_data)
            logger.info(f"Pushed tool event: {event_type} for {tool_name}")
            
        except Exception as e:
            logger.error(f"Failed to push tool event: {e}")

    def _inject_verification_info(self, event_data: dict, tool_name: str, processed_result: dict):
        """注入验证信息到事件数据中"""
        try:
            # 获取验证步骤
            steps = self._get_verification_steps(tool_name)
            
            # 从processed_result中提取URL和文件路径
            urls = []
            file_path = None
            
            if isinstance(processed_result, dict):
                # 从urls字段获取
                if "urls" in processed_result and isinstance(processed_result["urls"], list):
                    urls.extend(processed_result["urls"])
                # 从first_url字段获取
                if "first_url" in processed_result and processed_result["first_url"]:
                    urls.append(processed_result["first_url"])
                
                # 从file_path字段获取文件路径
                if "file_path" in processed_result and processed_result["file_path"]:
                    file_path = processed_result["file_path"]
            
            # 确保extra字段存在
            if "extra" not in event_data:
                event_data["extra"] = {}
            
            # 构建验证信息
            verification_info = {
                "steps": steps,
                "urls": urls
            }
            
            # 如果有文件路径，添加到验证信息中
            if file_path:
                verification_info["file_path"] = file_path
            
            # 注入验证信息
            event_data["extra"]["verification"] = verification_info
            
        except Exception as e:
            logger.error(f"Failed to inject verification info: {e}")

    def _get_tool_name_zh(self, tool_name: str) -> str:
        """获取工具名称的中文翻译"""
        tool_name_mapping = {
            # 搜索类工具
            "search_baidu": "百度搜索",
            "search_google": "谷歌搜索", 
            "tavily_search": "Tavily搜索",
            "search_wiki": "维基百科搜索",
            "image_search": "图片搜索",
            "audio_recognition": "音频识别",
            
            # 文件操作类工具
            "file_saver": "保存文件",
            "file_read": "读取文件",
            "file_write": "写入文件",
            "file_append": "追加文件",
            "file_delete": "删除文件",
            "file_list": "列出文件",
            "file_copy": "复制文件",
            "file_move": "移动文件",
            "file_str_replace": "文件内容替换",
            "file_find_in_content": "文件内容查找",
            
            # 代码执行类工具
            "execute_code": "代码执行器",
            "python_executor": "Python执行器",
            "shell_executor": "Shell执行器",
            
            # 网页操作类工具
            "web_scraper": "网页抓取",
            "web_navigator": "网页导航",
            "web_click": "网页点击",
            "web_input": "网页输入",
            "web_screenshot": "网页截图",
            "browser_use": "浏览器操作",
            "fetch_website_content": "抓取网页内容",
            
            # 图像分析类工具
            "image_analyzer": "图像分析",
            "image_ocr": "图像识别",
            "image_caption": "图像描述",
            
            # 视频分析类工具
            "video_analyzer": "视频分析",
            "video_extract": "视频提取",
            
            # 文档处理类工具
            "create_html_report": "创建HTML报告",
            "document_processor": "文档处理",
            "pdf_reader": "PDF阅读器",
            "word_processor": "Word处理",
            "excel_processor": "Excel处理",
            "extract_document_content": "抽取文档内容",
            
            # 数据库类工具
            "database_query": "数据库查询",
            "sql_executor": "SQL执行器",
            
            # 网络类工具
            "http_request": "HTTP请求",
            "api_call": "API调用",
            "webhook": "Webhook",
            
            # 计划管理类工具
            "create_plan": "创建计划",
            "update_plan": "更新计划",
            "mark_step": "标记步骤",
            "terminate": "终止任务",
            
            # 其他工具
            "calculator": "计算器",
            "translator": "翻译器",
            "summarizer": "摘要器",
            "text_analyzer": "文本分析",
            "data_processor": "数据处理",
            "chart_generator": "图表生成",
            "report_generator": "报告生成"
        }
        
        return tool_name_mapping.get(tool_name, tool_name)

    def reset_step_file_saver_count(self, step_index: int):
        """重置指定步骤的file_saver调用计数"""
        if step_index in self._file_saver_call_count:
            del self._file_saver_call_count[step_index]
            logger.info(f"Reset file_saver call count for step {step_index}")

    def _get_verification_steps(self, tool_name: str) -> list[str]:
        """基于工具名称获取验证步骤"""
        name = (tool_name or "").lower()
        
        # 搜索类：支持交叉验证
        if name in ("search_baidu", "search_google", "tavily_search", "search_wiki", "image_search"):
            return ["source_trace", "rule_assist", "self_consistency"]
        
        # 保存类
        if name in ("file_saver",):
            return ["source_trace"]
        
        # 文件处理类
        if name in ("file_read", "file_find_in_content", "file_str_replace"):
            return ["rule_assist"]
        
        # 代码执行/数据处理
        if name in ("execute_code",):
            return ["rule_assist", "self_consistency"]
        
        # 浏览抓取
        if name in ("browser_use", "fetch_website_content"):
            return ["source_trace", "rule_assist"]
        
        # 文档抽取
        if name in ("extract_document_content",):
            return ["rule_assist"]
        
        # 多模态/音频
        if name in ("ask_question_about_image", "ask_question_about_video", "audio_recognition"):
            return ["rule_assist", "self_consistency"]
        
        # 报告生成
        if name in ("create_html_report",):
            return ["rule_assist", "self_consistency"]
        
        # 未在清单中的工具：不返回任何步骤
        return []

    def execute(self, messages: List[Dict[str, Any]], step_index=None, max_iteration=10):  #调试修改的10
        for i in range(max_iteration):
            logger.info(f'act agent call with tools message: {messages}')
            
            # Use draft/verifier pattern if enabled and available
            if self.use_draft_verifier:
                if i == 0:  # Only print once per execution
                    print(f"\n🚀 DRAFT/VERIFIER MODE ENABLED (Agent: {self.agent_instance.instance_name}, Step: {step_index})")
                    print(f"   Draft Model: {self.draft_llm.model if self.draft_llm else 'None'}")
                    print(f"   Verifier Model: {self.verifier_llm.model if self.verifier_llm else 'None'}")
                
                logger.info(f'Using draft/verifier pattern for tool calls (iteration {i})')
                num_draft_candidates = int(os.environ.get("DRAFT_CANDIDATES", "3"))
                use_verifier = os.environ.get("USE_VERIFIER", "true").lower() == "true"
                response = self.llm.create_with_tools_draft_verifier(
                    messages, self.tools,
                    draft_llm=self.draft_llm,
                    verifier_llm=self.verifier_llm,
                    num_draft_candidates=num_draft_candidates,
                    use_verifier=use_verifier
                )
            else:
                response = self.llm.create_with_tools(messages, self.tools)
            
            logger.info(f'act agent call with tools response: {response}')

            # Process initial response
            result = self._process_response(response, messages, step_index)
            logger.info(f'iter {i} for {self.agent_instance.instance_name} call tools result: {result}')
            if result:
                return result

        if max_iteration > 1:
            return self._handle_max_iteration(messages, step_index)
        return messages[-1].get("content")

    def _process_response(self, response, messages, step_index):
        if not response.tool_calls:
            messages.append({"role": "assistant", "content": response.content})
            return response.content

        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls
        })

        results = self._execute_tool_calls(response.tool_calls, step_index)
        messages.extend(results)

        # Check for termination conditions
        for result in results:
            if result["name"] in ["terminate", "mark_step"]:
                return result["content"]
        return None

    def _execute_tool_calls(self, tool_calls, step_index):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                if function_name in self.functions:
                    futures.append(executor.submit(
                        self._execute_tool_call,
                        function_name=function_name,
                        function_args=function_args,
                        tool_call_id=tool_call.id,
                        step_index=step_index
                    ))
                else:
                    futures.append(executor.submit(
                        self._execute_mcp_tool_call,
                        function_name=function_name,
                        function_args=function_args,
                        tool_call_id=tool_call.id
                    ))

            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Unhandled exception: {e}", exc_info=True)
                    results.append({
                        "role": "tool",
                        "name": function_name,
                        "tool_call_id": tool_call.id,
                        "content": f"Execution error: {str(e)}"
                    })
        return results

    def _handle_max_iteration(self, messages, step_index):
        messages.append({"role": "user", "content": "Summarize the above conversation, use mark_step to mark the step"})
        mark_step_tools = [tool for tool in self.tools if tool['function']['name'] == 'mark_step']
        response = self.llm.create_with_tools(messages, mark_step_tools)

        result = self._process_response(response, messages, step_index)
        if result:
            return result

        return messages[-1].get("content")

    @time_record
    def _execute_tool_call(self, function_name="", function_args="", tool_call_id="", step_index=None):
        start_time = time.time()
        
        # 推送工具开始执行事件
        self._push_tool_event("tool_start", function_name, function_args, step_index=step_index)
        
        try:
            # Robust JSON arg parsing: tolerate code fences/None/empty/single quotes/trailing commas
            cleaned_args = function_args if function_args is not None else "{}"
            if isinstance(cleaned_args, bytes):
                cleaned_args = cleaned_args.decode('utf-8', errors='ignore')
            cleaned_args = str(cleaned_args).strip()
            # strip markdown fences
            if cleaned_args.startswith("```"):
                tmp = cleaned_args.strip('`')
                if "\n" in tmp:
                    tmp = tmp.split("\n", 1)[1]
                cleaned_args = tmp.strip()
                if cleaned_args.endswith("```"):
                    cleaned_args = cleaned_args[:-3]
            if cleaned_args == "" or cleaned_args.lower() in ("null", "none"):
                cleaned_args = "{}"
            try:
                args_dict = json.loads(cleaned_args)
            except Exception:
                repaired = cleaned_args.replace("'", '"').rstrip(',').strip()
                if repaired and not (repaired.startswith('{') or repaired.startswith('[')):
                    repaired = '{' + repaired + '}'
                try:
                    args_dict = json.loads(repaired)
                except Exception:
                    args_dict = {}

            if step_index is not None and 'step_index' not in args_dict and function_name in ['mark_step']:
                args_dict['step_index'] = step_index

            # 检查file_saver调用频率限制
            if function_name == 'file_saver' and step_index is not None:
                if step_index not in self._file_saver_call_count:
                    self._file_saver_call_count[step_index] = 0
                self._file_saver_call_count[step_index] += 1
                
                # 如果当前步骤已经调用过file_saver，给出警告并建议合并
                if self._file_saver_call_count[step_index] > 1:
                    logger.warning(f"file_saver called {self._file_saver_call_count[step_index]} times in step {step_index}. "
                                 f"Consider consolidating file saves to improve performance.")

            function_to_call = self.functions[function_name]

            # 检查是否是异步函数
            if inspect.iscoroutinefunction(function_to_call):
                # 创建新的事件循环来运行异步函数
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    # 归一化参数键（含函数名定制映射）
                    norm_args = self._normalize_tool_args(function_to_call, args_dict, function_name)
                    result = loop.run_until_complete(function_to_call(**norm_args))
                finally:
                    loop.close()
            else:
                # 同步函数直接调用
                norm_args = self._normalize_tool_args(function_to_call, args_dict, function_name)
                result = function_to_call(**norm_args)

            # 计算执行时间
            duration = time.time() - start_time
            
            # 推送工具执行完成事件
            self._push_tool_event("tool_complete", function_name, function_args, 
                                str(result), step_index, duration)

            # 记录工具调用信息到Plan对象（如果有plan引用且step_index有效）
            if self.plan and step_index is not None and hasattr(self.plan, 'add_tool_call'):
                try:
                    self.plan.add_tool_call(
                        step_index,
                        function_name,
                        function_args,
                        str(result),
                        duration=duration
                    )
                except Exception as e:
                    logger.warning(f"Failed to record tool call to plan: {e}")

            return {
                "role": "tool",
                "name": function_name,
                "content": str(result),
                "tool_call_id": tool_call_id
            }
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # 推送工具执行错误事件
            self._push_tool_event("tool_error", function_name, function_args, 
                                "", step_index, duration, error_msg)
            
            if self.plan and step_index is not None and hasattr(self.plan, 'add_tool_call'):
                try:
                    self.plan.add_tool_call(
                        step_index,
                        function_name,
                        function_args,
                        f"Execution error: {error_msg}",
                        duration=duration
                    )
                except Exception as record_err:
                    logger.warning(f"Failed to record errored tool call to plan: {record_err}")
            
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            return {
                "role": "tool",
                "name": function_name,
                "tool_call_id": tool_call_id,
                "content": f"Execution error: {str(e)}"
            }

    @time_record
    def _execute_mcp_tool_call(self, function_name="", function_args="", tool_call_id=""):
        start_time = time.time()
        
        # 推送MCP工具开始执行事件
        self._push_tool_event("tool_start", function_name, function_args, step_index=-1)
        
        loop = None
        try:
            mcp_tool, tool_name = self.find_mcp_tool(function_name)
            if mcp_tool and tool_name:
                cleaned_args = function_args.replace('\\\'', '\'')
                args_dict = json.loads(cleaned_args or "{}")
                # Windows系统需要特殊处理
                if sys.platform == "win32":
                    from asyncio import ProactorEventLoop
                    loop = ProactorEventLoop()
                else:
                    loop = asyncio.new_event_loop()

                asyncio.set_event_loop(loop)

                # 执行异步调用
                result = loop.run_until_complete(
                    MCPEngine.invoke_mcp_tool(
                        mcp_tool['mcp_name'],
                        mcp_tool['mcp_config'],
                        tool_name,
                        args_dict
                    )
                )
                
                # 计算执行时间
                duration = time.time() - start_time
                
                # 推送MCP工具执行完成事件
                self._push_tool_event("tool_complete", function_name, function_args, 
                                    str(result), -1, duration)
                
                # 记录MCP工具调用信息到Plan对象（如果有plan引用）
                if self.plan and hasattr(self.plan, 'add_tool_call'):
                    try:
                        # MCP工具调用没有step_index，使用-1表示全局工具调用
                        self.plan.add_tool_call(-1, function_name, function_args, str(result), duration=duration)
                    except Exception as e:
                        logger.warning(f"Failed to record MCP tool call to plan: {e}")
                
                return {
                    "role": "tool",
                    "name": function_name,
                    "content": str(result),
                    "tool_call_id": tool_call_id
                }
            else:
                duration = time.time() - start_time
                error_msg = f"Function {function_name} not found in available functions"
                
                # 推送MCP工具执行错误事件
                self._push_tool_event("tool_error", function_name, function_args, 
                                    "", -1, duration, error_msg)
                
                return {
                    "role": "tool",
                    "name": function_name,
                    "tool_call_id": tool_call_id,
                    "content": error_msg
                }
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # 推送MCP工具执行错误事件
            self._push_tool_event("tool_error", function_name, function_args, 
                                "", -1, duration, error_msg)
            
            if self.plan and hasattr(self.plan, 'add_tool_call'):
                try:
                    self.plan.add_tool_call(-1, function_name, function_args, f"Execution error: {error_msg}", duration=duration)
                except Exception as record_err:
                    logger.warning(f"Failed to record errored MCP tool call to plan: {record_err}")
            
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            return {
                "role": "tool",
                "name": function_name,
                "tool_call_id": tool_call_id,
                "content": f"Execution error: {str(e)}"
            }
        finally:
            # 清理事件循环
            if loop:
                loop.close()
                asyncio.set_event_loop(None)
