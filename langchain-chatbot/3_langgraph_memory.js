// Wraps LLM in a LangGraph workflow with persistent memory
import { ChatOpenAI } from "@langchain/openai";
import { v4 as uuidv4 } from "uuid";
import { START, END, MessagesAnnotation, StateGraph, MemorySaver } from "@langchain/langgraph";

const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

const callModel = async (state) => {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
};

const graph = new StateGraph(MessagesAnnotation).addNode("model", callModel).addEdge(START, "model").addEdge("model", END);

const app = graph.compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: uuidv4() } };

const input = { messages: [{ role: "user", content: "Hi! I'm Bob" }] };
const result = await app.invoke(input, config);
console.log("LangGraph Memory Output:", result.messages.at(-1).content);
