// Optional: Export the agent graph as a PNG image
import { writeFileSync } from "fs";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from "@langchain/langgraph";

const TAVILY_API_KEY = "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";
// --- Setup agent again ---
const tools = [new TavilySearch({ maxResults: 3, tavilyApiKey: TAVILY_API_KEY })];
const model = new ChatOpenAI({ temperature: 0 });
const memory = new MemorySaver();

const agent = createReactAgent({
  llm: model,
  tools,
  checkpointSaver: memory,
});

// --- Export graph image ---
const graph = agent.getGraph();
const image = await graph.drawMermaidPng();
const buffer = await image.arrayBuffer();

writeFileSync("./graphState.png", new Uint8Array(buffer));
console.log("Graph image saved to graphState.png");
