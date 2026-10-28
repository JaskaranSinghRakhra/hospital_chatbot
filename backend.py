
from langgraph.graph import StateGraph,START,END
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from retriever import (hospital_retriever,doctor_retriever,route_retriever,profile_retriever)
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()
api=os.getenv('NVIDIA_API_KEY')
model=ChatNVIDIA(
    api_key=api,
    model='nvidia/nemotron-3.5-lightning-30b-a3b',
    temperature=0,
    max_completion_tokens=1024
)

@tool
def hospital_info(query:str)->str:
    """Retrieve hospital services, facilities, departments, timings and policies."""
    docs=hospital_retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def doctor_search(query:str)->str:
    """Find suitable doctors based on symptoms, specialty and expertise."""
    docs=doctor_retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def hospital_route(query:str)->str:
    """Find hospital buildings, departments, floors, rooms and directions."""
    docs=route_retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def doctor_profile(query:str)->str:
    """Retrieve doctor qualifications, experience, specialization and patient feedback."""
    docs=profile_retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

tools=[hospital_info,doctor_search,hospital_route,doctor_profile]
llm_with_tools=model.bind_tools(tools)

class Chat(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

graph=StateGraph(Chat)

def query_extractor(state:Chat):
    prompt=SystemMessage(content="""You are the query router for a Hospital AI Assistant.
Use exactly one appropriate tool for the user's query.
hospital_info: General hospital services, facilities, departments, timings and policies.
doctor_search: Find a suitable doctor based on symptoms, specialty or expertise.
hospital_route: Find the location of a doctor, department, room, floor, building or facility.
doctor_profile: Qualifications, experience, specialization, expertise, languages or feedback about a specific doctor.
Use conversation history to understand follow-up queries.
Do not answer the user's question yourself. Call the appropriate tool.
If the intent is unclear, use hospital_info.""")
    response=llm_with_tools.invoke([prompt]+state["messages"])
    return {"messages":[response]}

def final_chat(state:Chat):
    prompt=SystemMessage(content="""You are Hospital AI Copilot.
Answer the user's query using the conversation and retrieved tool results.
Do not invent doctor details, locations or hospital policies.
If the information is insufficient, clearly say so.
Do not diagnose or prescribe treatment.
If emergency symptoms are described, advise seeking urgent medical assistance.""")
    response=model.invoke([prompt]+state["messages"])
    return {"messages":[response]}

graph.add_node("query_extractor",query_extractor)
graph.add_node("tools",ToolNode(tools,handle_tool_errors=True))
graph.add_node("final",final_chat)

graph.add_edge(START,"query_extractor")
graph.add_conditional_edges("query_extractor",tools_condition,{"tools":"tools",END:"final"})
graph.add_edge("tools","final")
graph.add_edge("final",END)

con=sqlite3.connect(database="chat_history.db",check_same_thread=False)
check=SqliteSaver(conn=con)
check.setup()
workflow=graph.compile(checkpointer=check)

def retrieve_all():
    threads=set()
    for checkpoint in check.list(None):
            threads.add(checkpoint.config['configurable']['thread_id'])
    return list(threads)
