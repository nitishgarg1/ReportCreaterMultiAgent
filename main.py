from src.agents.agents import build_search_agent
from rich import print

agent=build_search_agent()

res=agent.invoke({
    "messages":"give me latest info about ai"
})

print(res)