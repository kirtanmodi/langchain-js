// Basic ReAct Agent using LangGraph + OpenAI + Tavily
import { ChatOpenAI } from "@langchain/openai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { TavilySearch } from "@langchain/tavily";

const TAVILY_API_KEY = "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";

// --- AGENT SETUP ---
const tools = [new TavilySearch({ maxResults: 3, tavilyApiKey: TAVILY_API_KEY })];
const model = new ChatOpenAI({ temperature: 0 });
const memory = new MemorySaver();

const agent = createReactAgent({
  llm: model,
  tools,
  checkpointSaver: memory,
});

// --- USAGE ---
const threadId = "42"; // Can be any string

const state1 = await agent.invoke(
  {
    messages: [new HumanMessage("what is the current weather in sf")],
  },
  { configurable: { thread_id: threadId } }
);

console.log("Response 1:", state1.messages.at(-1).content);

const state2 = await agent.invoke(
  {
    messages: [new HumanMessage("what about ny")],
  },
  { configurable: { thread_id: threadId } }
);

console.log("Response 2:", state2.messages.at(-1).content);
