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
import json
import time
from json import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from app.agent_dispatcher.infrastructure.entity.exception.ZaeFrameworkException import ZaeFrameworkException
from app.cosight.task.time_record_util import time_record
from app.common.logger_util import logger
import asyncio

class ChatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, client: OpenAI, max_tokens: int = 4096,
                 temperature: float = 0.0, stream: bool = False, tools: List[Any] = None):
        self.tools = tools or []
        self.client = client
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def clean_none_values(data):
        """
        递归遍历数据结构，将所有 None 替换为 ""
        静态方法，无需实例化类即可调用
        """
        if isinstance(data, dict):
            return {k: ChatLLM.clean_none_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [ChatLLM.clean_none_values(item) for item in data]
        elif data is None:
            return ""
        else:
            return data

    @time_record
    def create_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict]):
        """
        Create a chat completion with support for function/tool calls
        """
        # 清洗提示词，去除None
        messages = ChatLLM.clean_none_values(messages)
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature
                )
                logger.info(f"LLM with tools chat completions response is {response}")
                if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                    self.check_and_fix_tool_call_params(response)
                elif hasattr(response, 'message') and response.message:
                    raise Exception(response.message)
                else:
                    raise Exception(response)
                break
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {json_error}")
                
                # 更详细的错误信息记录
                response_info = "No response object"
                if response is not None:
                    if hasattr(response, 'content'):
                        response_info = f"Response content: {response.content}"
                    elif hasattr(response, 'text'):
                        response_info = f"Response text: {response.text}"
                    elif hasattr(response, 'choices') and response.choices:
                        try:
                            content = response.choices[0].message.content if response.choices[0].message else "No message content"
                            response_info = f"Response message content: {content[:500]}..." if len(str(content)) > 500 else f"Response message content: {content}"
                        except Exception:
                            response_info = f"Response object: {type(response)} - {str(response)[:500]}..."
                    else:
                        response_info = f"Response object: {type(response)} - {str(response)[:500]}..."
                
                logger.error(f"Response details: {response_info}")
                
                if attempt == max_retries - 1:
                    raise ZaeFrameworkException(400, f"JSON decode error after {max_retries} attempts: {json_error}")
                time.sleep(5)  # 增加等待时间
            except Exception as e:
                logger.warning(f"chat with LLM error: {e} on attempt {attempt + 1}, retrying...", exc_info=True)
                if "TPM limit reached" in str(e):
                    time.sleep(60)
                elif "rate limit" in str(e).lower():
                    time.sleep(30)
                elif "timeout" in str(e).lower():
                    time.sleep(10)
                if attempt == max_retries-1:
                    logger.error(f"Failed to create after {max_retries} attempts.")
                    raise ZaeFrameworkException(400, f"chat with LLM failed, please check LLM config. reason：{e}")
                time.sleep(3)  # 增加等待时间，避免频繁重试

        if response and isinstance(response, ChatCompletion):
            # 去除think标签
            content = response.choices[0].message.content
            if content is not None and '</think>' in content:
                response.choices[0].message.content = content.split('</think>')[-1].strip('\n')
            return response.choices[0].message
        else:
            raise ZaeFrameworkException(400, f"chat with LLM failed, LLM response：{response}")

    def check_and_fix_tool_call_params(self, response):
        if response.choices[0].message.tool_calls:
            for attempt in range(3):
                try:
                    tool_call = response.choices[0].message.tool_calls[0].function
                    json.loads(tool_call.arguments)
                    break
                except JSONDecodeError as jsone:
                    logger.warning(f"Tool call arguments JSON decode error on attempt {attempt + 1}: {jsone}")
                    logger.warning(f"Invalid arguments: {tool_call.arguments}")
                    
                    try:
                        # 尝试修复JSON格式
                        fixed_arguments = self.chat_to_llm([{"role": "user",
                                                           "content": f"下面的json字符串格式有错误，请帮忙修正。重要：仅输出修正的字符串。\n{tool_call.arguments}"}])
                        # 验证修复后的JSON是否有效
                        json.loads(fixed_arguments)
                        tool_call.arguments = fixed_arguments
                        logger.info(f"Successfully fixed tool call arguments on attempt {attempt + 1}")
                        break
                    except Exception as fix_error:
                        logger.error(f"Failed to fix tool call arguments on attempt {attempt + 1}: {fix_error}")
                        if attempt == 2:  # 最后一次尝试
                            # 如果修复失败，使用默认的空JSON对象
                            tool_call.arguments = "{}"
                            logger.warning("Using empty JSON object as fallback for tool call arguments")
                            break

    @time_record
    def chat_to_llm(self, messages: List[Dict[str, Any]]):
        # 清洗提示词，去除None
        messages = ChatLLM.clean_none_values(messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        logger.info(f"LLM chat completions response is {response}")
        # 去除think标签
        content = response.choices[0].message.content
        if content is not None and '</think>' in content:
            response.choices[0].message.content = content.split('</think>')[-1].strip('\n')

        return response.choices[0].message.content

    @time_record
    def create_with_tools_draft_verifier(self, messages: List[Dict[str, Any]], tools: List[Dict],
                                         draft_llm: Optional['ChatLLM'] = None,
                                         verifier_llm: Optional['ChatLLM'] = None,
                                         num_draft_candidates: int = 1,
                                         use_verifier: bool = True):
        """
        Create tool calls using draft/verifier pattern for parallel generation and validation.
        
        Args:
            messages: Conversation messages
            tools: Available tools
            draft_llm: Draft model for generating candidate tool calls (faster/cheaper)
            verifier_llm: Verifier model for validating tool calls (more accurate)
            num_draft_candidates: Number of parallel draft candidates to generate
            use_verifier: Whether to use verifier model to validate candidates
        
        Returns:
            Message with verified tool calls
        """
        # 清洗提示词，去除None
        messages = ChatLLM.clean_none_values(messages)
        
        # If no draft/verifier models provided, fall back to standard method
        if not draft_llm:
            logger.info("No draft model provided, using standard create_with_tools")
            return self.create_with_tools(messages, tools)
        
        # Generate multiple draft candidates in parallel
        print("\n" + "="*80)
        print("🔵 DRAFT/VERIFIER MODE: Starting parallel tool call generation")
        print("="*80)
        logger.info(f"Generating {num_draft_candidates} draft tool call candidates in parallel")
        draft_candidates = self._generate_draft_candidates(
            messages, tools, draft_llm, num_draft_candidates
        )
        
        if not draft_candidates:
            print("⚠️  No draft candidates generated, falling back to standard method")
            logger.warning("No draft candidates generated, falling back to standard method")
            return self.create_with_tools(messages, tools)
        
        # If verifier is enabled and available, validate candidates
        if use_verifier and verifier_llm:
            print("\n" + "-"*80)
            print("🟢 VERIFIER: Validating draft candidates")
            print("-"*80)
            logger.info("Verifying draft candidates with verifier model")
            verified_response = self._verify_tool_calls(
                messages, tools, draft_candidates, verifier_llm
            )
            if verified_response:
                print("✅ VERIFIER: Selected verified tool calls")
                print("="*80 + "\n")
                return verified_response
        
        # If no verifier or verification failed, use the first draft candidate
        print("\n" + "-"*80)
        print("📋 SELECTING: Using best draft candidate (no verifier or verification failed)")
        print("-"*80)
        logger.info("Using best draft candidate (no verifier or verification failed)")
        best_candidate = self._select_best_candidate(draft_candidates)
        print("="*80 + "\n")
        return best_candidate

    # async def create_with_tools_draft_verifier(
    #     draft_model,
    #     verifier_model,
    #     tool_executor,
    #     initial_prompt: str,
    #     max_steps: int = 10,
    # ) -> Dict[str, Any]:
    #     """
    #     Parallel Draft + Verifier Execution Loop.

    #     - Draft model proposes next action (tool call or final answer).
    #     - If tool call is proposed:
    #         → Verifier runs in parallel.
    #         → If approved → tool is executed.
    #         → If rejected → continue asking draft for another prediction.
    #     - Loop stops when draft produces a final answer OR max_steps reached.
    #     """

    #     state = {"messages": [{"role": "user", "content": initial_prompt}]}

    #     async def run_draft(messages) -> Dict[str, Any]:
    #         """Request next action from draft model."""
    #         return await draft_model(messages)

    #     async def run_verifier(messages, draft_proposal) -> bool:
    #         """Ask verifier if draft proposal is valid."""
    #         verification_prompt = (
    #             f"Does this draft step look valid? Respond yes/no.\n\n{draft_proposal}"
    #         )
    #         v_msgs = messages + [{"role": "assistant", "content": draft_proposal},
    #                             {"role": "user", "content": verification_prompt}]
    #         return await verifier_model(v_msgs)

    #     for step in range(max_steps):

    #         # ➤ Step 1: Get next draft proposal
    #         draft_output = await run_draft(state["messages"])
    #         draft_msg = draft_output["content"]

    #         # If it's a final answer, return immediately
    #         if draft_output.get("is_final"):
    #             state["messages"].append({"role": "assistant", "content": draft_msg})
    #             return {"final": draft_msg, "steps": step + 1}

    #         # If it's a tool call: run verifier in parallel
    #         if draft_output.get("tool_call"):
    #             tool_call = draft_output["tool_call"]

    #             async with asyncio.TaskGroup() as tg:
    #                 verifier_task = tg.create_task(
    #                     run_verifier(state["messages"], draft_msg)
    #                 )

    #             verified = await verifier_task

    #             if verified:
    #                 # Approved → execute tool
    #                 tool_result = await tool_executor(tool_call["name"], tool_call["args"])
    #                 state["messages"].append({
    #                     "role": "tool",
    #                     "tool_name": tool_call["name"],
    #                     "content": tool_result,
    #                 })
    #             else:
    #                 # Rejected → append rejection note and continue drafting
    #                 state["messages"].append({
    #                     "role": "assistant",
    #                     "content": f"[Verifier rejected: {draft_msg}]"
    #                 })

    #         else:
    #             # Should not happen, but handle gracefully
    #             state["messages"].append({"role": "assistant", "content": draft_msg})

    #     return {"final": None, "error": "max_steps_exceeded", "steps": max_steps}


    def _generate_draft_candidates(self, messages: List[Dict[str, Any]], tools: List[Dict],
                                   draft_llm: 'ChatLLM', num_candidates: int) -> List[Any]:
        """
        Generate multiple draft tool call candidates in parallel.
        
        Returns:
            List of draft candidate responses
        """
        candidates = []
        
        candidate_index = [0]  # Use list to allow modification in nested function
        
        def generate_single_draft():
            idx = candidate_index[0]
            candidate_index[0] += 1
            try:
                print(f"  🔷 Draft Candidate {idx + 1}: Generating...")
                response = draft_llm.create_with_tools(messages, tools)
                
                # Print what tools this draft candidate predicted
                if response and response.tool_calls:
                    tool_names = [tc.function.name for tc in response.tool_calls]
                    print(f"  ✅ Draft Candidate {idx + 1}: Predicted {len(response.tool_calls)} tool(s) -> {', '.join(tool_names)}")
                    for tool_call in response.tool_calls:
                        print(f"      - {tool_call.function.name}({tool_call.function.arguments[:100]}...)")
                else:
                    content_preview = (response.content[:100] + "...") if response and response.content else "No content"
                    print(f"  📝 Draft Candidate {idx + 1}: No tool calls, content: {content_preview}")
                
                return response
            except Exception as e:
                print(f"  ❌ Draft Candidate {idx + 1}: Error - {str(e)[:100]}")
                logger.error(f"Error generating draft candidate: {e}", exc_info=True)
                return None
        
        # Generate candidates in parallel
        with ThreadPoolExecutor(max_workers=num_candidates) as executor:
            futures = [executor.submit(generate_single_draft) for _ in range(num_candidates)]
            
            for future in as_completed(futures):
                try:
                    candidate = future.result(timeout=120)  # 2 minute timeout per candidate
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    print(f"  ❌ Error getting draft candidate result: {str(e)[:100]}")
                    logger.error(f"Error getting draft candidate result: {e}", exc_info=True)
        
        print(f"\n📊 DRAFT SUMMARY: Generated {len(candidates)}/{num_candidates} successful candidates")
        logger.info(f"Generated {len(candidates)} draft candidates out of {num_candidates} attempts")
        return candidates

    def _verify_tool_calls(self, messages: List[Dict[str, Any]], tools: List[Dict],
                          draft_candidates: List[Any], verifier_llm: 'ChatLLM') -> Optional[Any]:
        """
        Use verifier model to validate and select the best tool calls from draft candidates.
        
        Returns:
            Verified response message, or None if verification fails
        """
        if not draft_candidates:
            return None
        
        # Prepare verification prompt
        verification_prompt = self._build_verification_prompt(messages, draft_candidates, tools)
        
        try:
            # Print summary of draft candidates for verification
            print("\n  📋 Draft Candidates Summary:")
            for i, candidate in enumerate(draft_candidates, 1):
                if candidate.tool_calls:
                    tool_names = [tc.function.name for tc in candidate.tool_calls]
                    print(f"    Candidate {i}: {len(candidate.tool_calls)} tool(s) -> {', '.join(tool_names)}")
                else:
                    print(f"    Candidate {i}: No tool calls")
            
            # Ask verifier to select/validate the best tool calls
            print("  🔍 Verifier: Analyzing candidates...")
            verification_messages = messages + [{"role": "user", "content": verification_prompt}]
            verified_response = verifier_llm.create_with_tools(verification_messages, tools)
            
            # If verifier returned tool calls, use them
            if verified_response and verified_response.tool_calls:
                tool_names = [tc.function.name for tc in verified_response.tool_calls]
                print(f"  ✅ Verifier: Selected {len(verified_response.tool_calls)} tool(s) -> {', '.join(tool_names)}")
                for tool_call in verified_response.tool_calls:
                    print(f"      - {tool_call.function.name}({tool_call.function.arguments[:100]}...)")
                logger.info(f"Verifier selected {len(verified_response.tool_calls)} tool calls")
                return verified_response
            
            # If verifier didn't return tool calls but we have draft candidates, use best draft
            print("  ⚠️  Verifier: Did not return tool calls, using best draft candidate")
            logger.info("Verifier did not return tool calls, using best draft candidate")
            return self._select_best_candidate(draft_candidates)
            
        except Exception as e:
            print(f"  ❌ Verifier: Error during verification - {str(e)[:100]}")
            logger.error(f"Error during verification: {e}", exc_info=True)
            return None

    def _build_verification_prompt(self, messages: List[Dict[str, Any]], 
                                   draft_candidates: List[Any], tools: List[Dict]) -> str:
        """
        Build a prompt for the verifier model to evaluate draft candidates.
        """
        prompt_parts = [
            "You are a tool call verifier. Below are multiple candidate tool calls generated by a draft model.",
            "Your task is to select the best tool calls or generate improved ones based on the context.\n"
        ]
        
        # Add draft candidates
        for i, candidate in enumerate(draft_candidates, 1):
            prompt_parts.append(f"\n--- Draft Candidate {i} ---")
            if candidate.tool_calls:
                for tool_call in candidate.tool_calls:
                    prompt_parts.append(
                        f"Tool: {tool_call.function.name}\n"
                        f"Arguments: {tool_call.function.arguments}"
                    )
            else:
                prompt_parts.append(f"Content: {candidate.content}")
        
        prompt_parts.append(
            "\nPlease analyze these candidates and provide the best tool calls for the task. "
            "You can select from the candidates, combine them, or generate new ones if needed."
        )
        
        return "\n".join(prompt_parts)

    def _select_best_candidate(self, candidates: List[Any]) -> Any:
        """
        Select the best candidate from draft candidates.
        Currently selects the first candidate with tool calls, or the first candidate if none have tool calls.
        """
        if not candidates:
            raise ValueError("No candidates to select from")
        
        # Prefer candidates with tool calls
        candidates_with_tools = [c for c in candidates if c.tool_calls]
        if candidates_with_tools:
            # Select the candidate with the most tool calls (most comprehensive)
            best = max(candidates_with_tools, key=lambda c: len(c.tool_calls) if c.tool_calls else 0)
            tool_names = [tc.function.name for tc in best.tool_calls] if best.tool_calls else []
            print(f"  🎯 SELECTED: Best candidate with {len(best.tool_calls) if best.tool_calls else 0} tool(s) -> {', '.join(tool_names)}")
            for tool_call in (best.tool_calls or []):
                print(f"      - {tool_call.function.name}({tool_call.function.arguments[:100]}...)")
            return best
        
        # If no candidates have tool calls, return the first one
        print(f"  📝 SELECTED: First candidate (no tool calls in any candidate)")
        return candidates[0]
