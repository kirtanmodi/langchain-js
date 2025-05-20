// Full custom ReAct agent with visible logic and conditional flows

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { TavilySearch } from "@langchain/tavily";

const TAVILY_API_KEY = "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";

// --- Setup tools + model ---
const tools = [new TavilySearch({ maxResults: 3, tavilyApiKey: TAVILY_API_KEY })];
const toolNode = new ToolNode(tools);

const model = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
}).bindTools(tools); // Gives the model access to the tools

// --- Decision logic ---
function shouldContinue({ messages }) {
  const last = messages[messages.length - 1];
  if (last instanceof AIMessage && last.tool_calls?.length) return "tools";
  return "__end__";
}

// --- Call the model ---
async function callModel({ messages }) {
  const response = await model.invoke(messages);
  return { messages: [response] };
}

// --- Create the graph ---
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue);

const app = workflow.compile();

// --- Use it ---
const state1 = await app.invoke({
  messages: [new HumanMessage("what is the weather in sf")],
});
console.log("SF:", state1.messages.at(-1).content);

const state2 = await app.invoke({
  messages: [...state1.messages, new HumanMessage("what about surat, india?")],
});
console.log("NY:", state2.messages.at(-1).content);
