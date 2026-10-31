import asyncio
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

SERVERS = {
    "ExpenseTracker":{
        "transport":"stdio",
        "command":sys.executable,
        "args":["-c","from multi_agent_project import main; main()"],
    }
}
async def run_agent(prompt):
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    named_tools = {tool.name: tool for tool in tools}

    llm = ChatOpenAI(model="gpt-5")
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(prompt)]
    response = await llm_with_tools.ainvoke(messages)

    # Keep going until the model stops asking for tools (it may need several
    # rounds, e.g. add_expense first and then list_expenses)
    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            tool_message = await named_tools[tool_call["name"]].ainvoke(tool_call)
            messages.append(tool_message)
        response = await llm_with_tools.ainvoke(messages)

    return response.content


async def main():
    prompt = "add an expense of 435 of cab and then show me all expenses."
    print(f"Final Response: {await run_agent(prompt)}")

if __name__ == "__main__":
    asyncio.run(main())