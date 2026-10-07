import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp import MCPToolkit
from deepagents import create_deep_agent
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

load_dotenv()

async def run_agent():
    print("Initializing Qwen model...")
    qwen_llm = ChatOpenAI(
        model="qwen36-35b",
        api_key=os.getenv("OPENWEBUI_API_KEY"),
        base_url=os.getenv("OPENWEBUI_BASE_URL"),
        max_tokens=2048,
    )

    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    print(f"Connecting to Grafana MCP at {mcp_url}...")

    async with streamablehttp_client(mcp_url) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            grafana_toolkit = MCPToolkit(session=session, server_url=mcp_url, transport="streamable-http")
            await grafana_toolkit.initialize()
            grafana_tools = grafana_toolkit.get_tools()

            print(f"Successfully loaded {len(grafana_tools)} tools from Grafana.")

            system_prompt = """You are the CCAST AI Monitoring Assistant. Your job is to help the systems team query
and interpret telemetry data across our HPC clusters using Grafana.

When asked about infrastructure:
1. Use your Grafana tools to fetch relevant dashboards or metrics.
2. Analyze the time-series data.
3. Provide a clear summary of the hardware state."""

            monitoring_agent = create_deep_agent(
                model=qwen_llm,
                tools=grafana_tools
            )

            print("Sending test query to agent...")
            test_prompt = "What dashboards are available to monitor node health and Infiniband switches?"
            combined_prompt = f"{system_prompt}\n\nTask: {test_prompt}"

            response = await monitoring_agent.ainvoke({
                "messages": [{"role": "user", "content": combined_prompt}]
            })

            print("\n=== Agent Response ===")
            print(response["messages"][-1]["content"])
            print("========================\n")

if __name__ == "__main__":
    asyncio.run(run_agent())