// 5_orchestrator_multi_agents.js
// A multi-agent system using LangGraph where a Coordinator routes tasks to specialized agents

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { TavilySearch } from "@langchain/tavily";
import { Calculator } from "@langchain/community/tools/calculator";

// Your Tavily API key (never hardcode this in production)
const TAVILY_API_KEY = "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";

// Define constants for agent names/types
const AGENT_TYPES = {
  RESEARCHER: "researcher",
  CALCULATOR: "calculator",
  COORDINATOR: "coordinator",
};

// Set up individual tools for research and calculator agents
const researchTools = [new TavilySearch({ maxResults: 3, tavilyApiKey: TAVILY_API_KEY })];
const calculatorTools = [new Calculator()];

// Create tool nodes for tool usage
const researchToolNode = new ToolNode(researchTools);
const calculatorToolNode = new ToolNode(calculatorTools);

// Create dedicated LLM models for each agent and bind their tools
const researcherModel = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 }).bindTools(researchTools);
const calculatorModel = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 }).bindTools(calculatorTools);
const coordinatorModel = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

// Agent-specific system prompts to guide their behavior
const researcherSystemPrompt = new SystemMessage(
  "You are a research specialist. Your job is to find information to answer user questions. Be thorough and cite your sources."
);

const calculatorSystemPrompt = new SystemMessage(
  "You are a math specialist. Your job is to solve mathematical problems. Show your work step-by-step."
);

const coordinatorSystemPrompt = new SystemMessage(
  "You are a coordinator agent. Your job is to:\n" +
    "1. Analyze the user's question\n" +
    "2. Decide which specialist agent should handle it:\n" +
    "   - RESEARCHER for factual questions requiring internet search\n" +
    "   - CALCULATOR for mathematical questions\n" +
    "3. If the question is fully answered, provide FINAL_ANSWER\n" +
    "Always respond with exactly one of: RESEARCHER, CALCULATOR, or FINAL_ANSWER:[your final answer here]"
);

// Research agent logic: prepend system prompt and invoke model
async function callResearcherAgent({ messages }) {
  const agentMessages = [researcherSystemPrompt, ...messages];
  const response = await researcherModel.invoke(agentMessages);
  return { messages: [response] };
}

// Calculator agent logic: prepend system prompt and invoke model
async function callCalculatorAgent({ messages }) {
  const agentMessages = [calculatorSystemPrompt, ...messages];
  const response = await calculatorModel.invoke(agentMessages);
  return { messages: [response] };
}

// Coordinator agent logic: decides what to do next or gives final answer
async function callCoordinatorAgent({ messages }) {
  const agentMessages = [coordinatorSystemPrompt, ...messages];
  const response = await coordinatorModel.invoke(agentMessages);
  return { messages: [response] };
}

// Define the function that determines whether to continue or not
function shouldContinue({ messages }) {
  const lastMessage = messages[messages.length - 1];

  if (lastMessage instanceof AIMessage) {
    const content = lastMessage.content;

    if (content.includes("FINAL_ANSWER:")) {
      return "end";
    } else if (content.includes(AGENT_TYPES.RESEARCHER)) {
      return "researcher";
    } else if (content.includes(AGENT_TYPES.CALCULATOR)) {
      return "calculator";
    }
  }

  // Default to coordinator if can't determine
  return "coordinator";
}

// Create a new graph with MessagesAnnotation
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("coordinator", callCoordinatorAgent)
  .addNode("researcher", callResearcherAgent)
  .addNode("calculator", callCalculatorAgent)
  .addNode("researchTools", researchToolNode)
  .addNode("calculatorTools", calculatorToolNode)

  // Define the entry point
  .addEdge("__start__", "coordinator")

  // Define conditional edges from coordinator
  .addConditionalEdges("coordinator", shouldContinue, {
    researcher: "researcher",
    calculator: "calculator",
    end: "__end__",
    coordinator: "coordinator",
  })

  // Define tool routing for researcher
  .addConditionalEdges(
    "researcher",
    ({ messages }) => {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
        return "researchTools";
      }
      return "coordinator";
    },
    {
      researchTools: "researchTools",
      coordinator: "coordinator",
    }
  )

  // Define tool routing for calculator
  .addConditionalEdges(
    "calculator",
    ({ messages }) => {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
        return "calculatorTools";
      }
      return "coordinator";
    },
    {
      calculatorTools: "calculatorTools",
      coordinator: "coordinator",
    }
  )

  // Connect tools back to their agents
  .addEdge("researchTools", "researcher")
  .addEdge("calculatorTools", "calculator");

// Compile the graph into a runnable application
const app = workflow.compile();

// Example usage with different types of queries
async function main() {
  console.log("Starting multi-agent system...");

  // Research-based question
  console.log("\nAsking a research question...");
  const researchState = await app.invoke({
    messages: [new HumanMessage("What are the latest developments in quantum computing?")],
  });
  console.log("\nResearch Question Result:");
  console.log(researchState.messages[researchState.messages.length - 1].content);

  // Math problem
  console.log("\nAsking a math question...");
  const mathState = await app.invoke({
    messages: [new HumanMessage("If I have 145 apples and give away 27% of them, how many do I have left?")],
  });
  console.log("\nMath Question Result:");
  console.log(mathState.messages[mathState.messages.length - 1].content);

  // Mixed query that might require both agents
  console.log("\nAsking a complex question requiring multiple agents...");
  const mixedState = await app.invoke({
    messages: [new HumanMessage("What is the GDP of France in 2023 and how does it compare to 15% of the US GDP?")],
  });
  console.log("\nMixed Question Result:");
  console.log(mixedState.messages[mixedState.messages.length - 1].content);
}

// Run the main function
main().catch(console.error);
