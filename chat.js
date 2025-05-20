import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

const result = await llm.invoke([{ role: "user", content: "Hi I'm Kirtan" }]);

console.log(result);
