// 9_plugin_system_for_tools.js
// A flexible plugin system for dynamically adding tools to a LangGraph agent

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { TavilySearch } from "@langchain/tavily";
import { Calculator } from "@langchain/community/tools/calculator";
import { DynamicTool } from "@langchain/core/tools";
import { z } from "zod";

// Placeholder for Tavily API key - in production, get this from environment variables
const TAVILY_API_KEY = "your-tavily-api-key";

/**
 * PluginSystem - A class to manage and dynamically load tools for LangGraph agents
 */
class PluginSystem {
  constructor() {
    this.plugins = new Map();
    this.toolNodes = new Map();
    this.llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      temperature: 0,
    });
  }

  /**
   * Register a plugin with the system
   * @param {string} name - Unique plugin identifier
   * @param {Object} plugin - The plugin tool object
   * @param {string} description - Human-readable description of the plugin
   */
  registerPlugin(name, plugin, description) {
    if (this.plugins.has(name)) {
      console.warn(`Plugin ${name} already registered. Overwriting...`);
    }

    this.plugins.set(name, {
      tool: plugin,
      description,
      enabled: true,
    });

    // Create a corresponding tool node
    this.toolNodes.set(name, new ToolNode([plugin]));

    console.log(`Plugin registered: ${name} - ${description}`);
    return this;
  }

  /**
   * Unregister a plugin from the system
   * @param {string} name - Plugin identifier to remove
   */
  unregisterPlugin(name) {
    if (!this.plugins.has(name)) {
      console.warn(`Plugin ${name} not found, nothing to unregister.`);
      return false;
    }

    this.plugins.delete(name);
    this.toolNodes.delete(name);
    console.log(`Plugin unregistered: ${name}`);
    return true;
  }

  /**
   * Enable or disable a plugin
   * @param {string} name - Plugin identifier
   * @param {boolean} enabled - Whether to enable or disable
   */
  setPluginEnabled(name, enabled) {
    if (!this.plugins.has(name)) {
      console.warn(`Plugin ${name} not found, can't change status.`);
      return false;
    }

    const plugin = this.plugins.get(name);
    plugin.enabled = enabled;
    this.plugins.set(name, plugin);

    console.log(`Plugin ${name} ${enabled ? "enabled" : "disabled"}`);
    return true;
  }

  /**
   * Get all registered plugins
   * @returns {Array} Array of plugin objects
   */
  getAllPlugins() {
    return Array.from(this.plugins.entries()).map(([name, info]) => ({
      name,
      description: info.description,
      enabled: info.enabled,
    }));
  }

  /**
   * Get all enabled plugins
   * @returns {Array} Array of enabled plugin tools
   */
  getEnabledPlugins() {
    return Array.from(this.plugins.entries())
      .filter(([, info]) => info.enabled)
      .map(([name, info]) => ({
        name,
        tool: info.tool,
        description: info.description,
      }));
  }

  /**
   * Get enabled plugin tools
   * @returns {Array} Array of enabled plugin tool objects
   */
  getEnabledTools() {
    return this.getEnabledPlugins().map((plugin) => plugin.tool);
  }

  /**
   * Get a tool node for a specific plugin
   * @param {string} name - Plugin identifier
   * @returns {ToolNode} The tool node for the plugin
   */
  getToolNode(name) {
    return this.toolNodes.get(name);
  }

  /**
   * Get all enabled tool nodes
   * @returns {Object} Map of tool node names to tool nodes
   */
  getEnabledToolNodes() {
    return new Map(Array.from(this.toolNodes.entries()).filter(([name]) => this.plugins.get(name)?.enabled));
  }

  /**
   * Create a custom tool from a function
   * @param {string} name - Tool name
   * @param {string} description - Tool description
   * @param {Function} func - Function to execute when tool is called
   * @param {Object} schema - Zod schema for the tool parameters
   * @returns {DynamicTool} A dynamic tool
   */
  createCustomTool(name, description, func, schema = null) {
    let tool;

    if (schema) {
      // Create a tool with schema validation
      tool = new DynamicTool({
        name,
        description,
        func,
        schema,
      });
    } else {
      // Create a simple tool without schema
      tool = new DynamicTool({
        name,
        description,
        func,
      });
    }

    return tool;
  }
}

/**
 * AgentWithPlugins - A LangGraph agent with plugin support
 */
class AgentWithPlugins {
  constructor(systemPrompt) {
    this.pluginSystem = new PluginSystem();
    this.systemPrompt = systemPrompt || this.getDefaultSystemPrompt();
    this.llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      temperature: 0,
    });
    this.graph = null;
  }

  /**
   * Get the default system prompt
   * @returns {string} Default system prompt
   */
  getDefaultSystemPrompt() {
    return `You are a helpful AI assistant that can use various tools to help the user.
You have access to the following tools:
{tool_descriptions}

Always use tools when they can help answer the user's question better.
After using a tool, analyze the result and determine if you need another tool or can provide the final answer.
Be concise and helpful in your responses.`;
  }

  /**
   * Register built-in plugins
   */
  registerBuiltInPlugins() {
    // Register search capability
    const searchTool = new TavilySearch({
      maxResults: 3,
      tavilyApiKey: TAVILY_API_KEY,
    });

    this.pluginSystem.registerPlugin("search", searchTool, "Search the web for current information");

    // Register calculator
    const calculatorTool = new Calculator();

    this.pluginSystem.registerPlugin("calculator", calculatorTool, "Perform mathematical calculations");

    // Register a time tool
    const timeTool = this.pluginSystem.createCustomTool("current_time", "Get the current date and time", async () => {
      const now = new Date();
      return now.toLocaleString();
    });

    this.pluginSystem.registerPlugin("time", timeTool, "Get the current time and date");

    return this;
  }

  /**
   * Register a custom plugin
   * @param {string} name - Plugin name
   * @param {Object} tool - Tool object
   * @param {string} description - Description of the tool
   */
  registerPlugin(name, tool, description) {
    this.pluginSystem.registerPlugin(name, tool, description);
    return this;
  }

  /**
   * Register a custom function as a plugin
   * @param {string} name - Plugin name
   * @param {string} description - Description of the plugin
   * @param {Function} func - Function to execute
   * @param {Object} schema - Optional Zod schema for parameters
   */
  registerFunctionPlugin(name, description, func, schema = null) {
    const tool = this.pluginSystem.createCustomTool(name, description, func, schema);
    this.pluginSystem.registerPlugin(name, tool, description);
    return this;
  }

  /**
   * Build the tool descriptions for the system prompt
   * @returns {string} Formatted tool descriptions
   */
  buildToolDescriptions() {
    const enabledPlugins = this.pluginSystem.getEnabledPlugins();

    return enabledPlugins.map((plugin) => `- ${plugin.name}: ${plugin.description}`).join("\n");
  }

  /**
   * Update the system prompt with tool descriptions
   * @returns {string} Updated system prompt
   */
  getUpdatedSystemPrompt() {
    const toolDescriptions = this.buildToolDescriptions();
    return this.systemPrompt.replace("{tool_descriptions}", toolDescriptions);
  }

  /**
   * Initialize the agent with bindings
   */
  async initialize() {
    // Create LLM with tools bound
    const tools = this.pluginSystem.getEnabledTools();
    this.boundLLM = this.llm.bindTools(tools);

    // Create the graph
    this.buildGraph();

    return this;
  }

  /**
   * Build the agent graph
   */
  buildGraph() {
    // Create graph nodes for agent and tools
    const workflow = new StateGraph(MessagesAnnotation);

    // Add agent node
    workflow.addNode("agent", this.agentNode.bind(this));

    // Add tool nodes from plugins
    const toolNodes = this.pluginSystem.getEnabledToolNodes();
    for (const [name, node] of toolNodes.entries()) {
      workflow.addNode(name, node);
    }

    // Add edges
    workflow.addEdge("__start__", "agent");

    // Add conditional routing from agent to tools or end
    workflow.addConditionalEdges("agent", this.routeFromAgent.bind(this), {
      __end__: "__end__",
      ...Object.fromEntries(Array.from(toolNodes.keys()).map((name) => [name, name])),
    });

    // Add edges from tools back to agent
    for (const name of toolNodes.keys()) {
      workflow.addEdge(name, "agent");
    }

    // Compile the graph with a higher recursion limit
    this.graph = workflow.compile({
      recursionLimit: 100, // Increase from default 25
    });
  }

  /**
   * Agent node for processing messages
   * @param {Object} state - Current state object
   * @returns {Object} Updated state
   */
  async agentNode({ messages }) {
    // Prepend system message on first run
    if (!messages.some((m) => m instanceof SystemMessage)) {
      const systemMessage = new SystemMessage(this.getUpdatedSystemPrompt());
      messages = [systemMessage, ...messages];
    }

    try {
      // Invoke the language model with tools
      const response = await this.boundLLM.invoke(messages);

      // Return updated messages
      return { messages: [...messages, response] };
    } catch (error) {
      console.error("Error invoking LLM:", error.message);
      // Return a fallback message if the LLM invocation fails
      return {
        messages: [
          ...messages,
          new AIMessage("I encountered an error while processing your request. Could you please try again or rephrase your question?"),
        ],
      };
    }
  }

  /**
   * Route from agent node based on the presence of tool calls
   * @param {Object} state - Current state object
   * @returns {string} Next node to route to
   */
  routeFromAgent({ messages }) {
    const lastMessage = messages[messages.length - 1];

    if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
      // Get the tool name from the first tool call
      const toolCall = lastMessage.tool_calls[0];
      const toolName = toolCall.name;

      // Check if this tool is registered and enabled
      if (this.pluginSystem.getEnabledPlugins().some((p) => p.name === toolName)) {
        console.log(`Routing to tool: ${toolName}`);
        return toolName;
      }
    }

    // Check if we're in a potential infinite loop
    const responseCount = messages.filter((m) => m instanceof AIMessage).length;
    if (responseCount > 10) {
      console.warn("Potential infinite loop detected - forcing end of conversation");
      return "__end__";
    }

    // If no tool calls or unknown tool, end the conversation
    return "__end__";
  }

  /**
   * Run the agent with a user message
   * @param {string} message - User message
   * @returns {Object} Final state with all messages
   */
  async run(message) {
    if (!this.graph) {
      await this.initialize();
    }

    // Create a new state with just the user message
    const initialState = {
      messages: [new HumanMessage(message)],
    };

    try {
      // Invoke the graph
      return await this.graph.invoke(initialState);
    } catch (error) {
      console.error("Error running agent:", error.message);

      // Return a simplified response if the graph execution fails
      return {
        messages: [
          new HumanMessage(message),
          new AIMessage(
            `I encountered an error while processing your request: ${error.message}. This might be due to a tool misbehaving or a recursion issue.`
          ),
        ],
      };
    }
  }
}

/**
 * Create example plugins for demonstration
 */
async function createExamplePlugins() {
  // Create a weather plugin
  const weatherTool = new DynamicTool({
    name: "get_weather",
    description: "Get the current weather for a location",
    func: async (location) => {
      try {
        // Simulate API call
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Return mock weather data
        return JSON.stringify({
          location,
          temperature: Math.floor(Math.random() * 30) + 5,
          condition: ["sunny", "cloudy", "rainy", "snowy"][Math.floor(Math.random() * 4)],
          humidity: Math.floor(Math.random() * 60) + 30,
        });
      } catch (error) {
        console.error("Weather tool error:", error);
        return JSON.stringify({ error: "Failed to get weather data" });
      }
    },
    schema: z.string().describe("The city and country, e.g. 'London, UK'"),
  });

  // Create a translation plugin
  const translateTool = new DynamicTool({
    name: "translate",
    description: "Translate text from one language to another",
    func: async (input) => {
      try {
        const params = JSON.parse(input);
        const { text, target_language } = params;

        if (!text || !target_language) {
          return "Error: Both 'text' and 'target_language' are required";
        }

        // Simulate translation API call
        await new Promise((resolve) => setTimeout(resolve, 300));

        // Return mock translation (just append the target language)
        return `Translated to ${target_language}: ${text} [translation simulation]`;
      } catch (error) {
        console.error("Translation tool error:", error);
        return "Error: Please provide valid JSON input with 'text' and 'target_language' fields";
      }
    },
    schema: z
      .object({
        text: z.string().describe("The text to translate"),
        target_language: z.string().describe("The target language code or name"),
      })
      .describe("Parameters for translation"),
  });

  // Create a simple note-taking plugin
  let notes = [];

  const notepadTool = new DynamicTool({
    name: "notepad",
    description: "Save or retrieve notes",
    func: async (input) => {
      try {
        const params = JSON.parse(input);

        if (params.action === "save") {
          if (!params.text) {
            return "Error: 'text' field is required for saving notes";
          }

          notes.push({
            id: notes.length + 1,
            text: params.text,
            timestamp: new Date().toISOString(),
          });
          return `Note saved with ID ${notes.length}`;
        } else if (params.action === "get") {
          if (params.id) {
            const note = notes.find((n) => n.id === params.id);
            return note ? JSON.stringify(note) : "Note not found";
          } else {
            return JSON.stringify(notes);
          }
        } else {
          return "Unknown action. Use 'save' or 'get'.";
        }
      } catch (error) {
        console.error("Notepad tool error:", error);
        return "Error: Please provide valid JSON input with 'action' field";
      }
    },
    schema: z
      .object({
        action: z.enum(["save", "get"]).describe("The action to perform"),
        text: z.string().optional().describe("The note text to save"),
        id: z.number().optional().describe("The note ID to retrieve"),
      })
      .describe("Parameters for notepad operations"),
  });

  return {
    weatherTool,
    translateTool,
    notepadTool,
  };
}

// Simplified demo function to reduce chance of tool recursion issues
async function main() {
  console.log("🔌 Initializing Plugin System...");

  // Create agent with plugins
  const agent = new AgentWithPlugins();

  // Register built-in plugins
  agent.registerBuiltInPlugins();

  // Create and register example plugins
  const examplePlugins = await createExamplePlugins();
  agent.registerPlugin("weather", examplePlugins.weatherTool, "Get weather information for a location");
  agent.registerPlugin("translate", examplePlugins.translateTool, "Translate text between languages");
  agent.registerPlugin("notepad", examplePlugins.notepadTool, "Save and retrieve notes");

  // Register a simple custom function plugin
  agent.registerFunctionPlugin(
    "random_number",
    "Generate a random number between min and max",
    async (input) => {
      try {
        // Handle different possible input formats
        let min = 1;
        let max = 100;

        // If input is a string that looks like "10,50"
        if (typeof input === "string" && input.includes(",")) {
          const parts = input.split(",").map((part) => parseInt(part.trim(), 10));
          if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
            [min, max] = parts;
          }
        }
        // If input is a JSON string with min/max properties
        else if (typeof input === "string" && (input.includes("{") || input.trim() === "{}")) {
          try {
            const params = JSON.parse(input);
            min = params.min || min;
            max = params.max || max;
          } catch (e) {
            console.error("Could not parse JSON, using default range", e);
            console.log("Could not parse JSON, using default range");
          }
        }

        // Ensure min is less than max
        if (min > max) {
          [min, max] = [max, min];
        }

        // Generate and return the random number
        const randomNum = Math.floor(Math.random() * (max - min + 1)) + min;
        return randomNum.toString();
      } catch (error) {
        console.error("Random number tool error:", error);
        // Fallback to a simple random number between 1-100
        return Math.floor(Math.random() * 100 + 1).toString();
      }
    },
    z.any().describe("Min and max values for the random number. Can be provided as 'min,max' or as a JSON object with min and max properties.")
  );

  // Initialize the agent
  await agent.initialize();

  // Example conversations to demonstrate the plugin system
  const examples = [
    "What's the weather like in Tokyo?",
    "Calculate 15% of 67.80",
    "What time is it now?",
    "Generate a random number between 10 and 50",
  ];

  // Process each example query
  for (const example of examples) {
    console.log("\n---------------------------------------------");
    console.log(`👤 User: ${example}`);

    try {
      const result = await agent.run(example);
      const messages = result.messages;

      // Print only AI responses (skip system message)
      const responses = messages.filter((m) => m instanceof AIMessage);
      for (const response of responses) {
        if (response.tool_calls?.length) {
          console.log(`🔧 Tool used: ${response.tool_calls[0].name}`);
          console.log(`🔧 Tool input: ${JSON.stringify(response.tool_calls[0].args)}`);
        } else {
          console.log(`🤖 Assistant: ${response.content}`);
        }
      }
    } catch (error) {
      console.error(`❌ Error processing example "${example}":`, error.message);
    }
  }

  console.log("\n---------------------------------------------");
  console.log("✅ Plugin System Demonstration Complete");
}

// Run the demonstration
main().catch(console.error);
