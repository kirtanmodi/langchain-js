// Demonstrates 4 agents with 3 custom tools (no 3rd-party API)
// Uses LangGraph and LangChain core only

import { ChatOpenAI } from "@langchain/openai";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";

// -----------------------------
// Define Custom Tools
// -----------------------------

// JokeTool returns a static joke
const JokeTool = new DynamicStructuredTool({
  name: "joke_tool",
  description: "Returns a funny joke",
  schema: z.object({}),
  func: async () => "Why did the developer go broke? Because he used up all his cache.",
});

// DateTool returns the current date
const DateTool = new DynamicStructuredTool({
  name: "date_tool",
  description: "Returns the current date",
  schema: z.object({}),
  func: async () => `Today is ${new Date().toDateString()}`,
});

// EchoTool repeats input text
const EchoTool = new DynamicStructuredTool({
  name: "echo_tool",
  description: "Echoes the user's input",
  schema: z.object({ input: z.string() }),
  func: async ({ input }) => `You said: "${input}"`,
});

// -----------------------------
// Shared Model + Memory
// -----------------------------

const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const memory = new MemorySaver();

// -----------------------------
// Create Agents with Different Toolsets
// -----------------------------

const agent1 = createReactAgent({ llm, tools: [JokeTool], checkpointSaver: memory });
const agent2 = createReactAgent({ llm, tools: [DateTool], checkpointSaver: memory });
const agent3 = createReactAgent({ llm, tools: [EchoTool], checkpointSaver: memory });
const agent4 = createReactAgent({ llm, tools: [JokeTool, DateTool, EchoTool], checkpointSaver: memory });

// -----------------------------
// Utility to Run Agents
// -----------------------------

const runAgent = async (agent, question, threadId) => {
  const result = await agent.invoke({ messages: [new HumanMessage(question)] }, { configurable: { thread_id: threadId } });
  console.log(`\n[${threadId}]`, result.messages.at(-1).content);
};

// -----------------------------
// Run Examples
// -----------------------------

await runAgent(agent1, "Tell me a joke", "agent1");
await runAgent(agent2, "What day is it?", "agent2");
await runAgent(agent3, "Repeat this: I love LangGraph", "agent3");
await runAgent(agent4, "What day is it? Then tell a joke and repeat: Hello!", "agent4");
