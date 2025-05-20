// Basic LLM call without any prompt or memory
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

const response = await llm.invoke([{ role: "user", content: "Hi, I'm Bob" }]);
console.log("LLM Output:", response.content);
