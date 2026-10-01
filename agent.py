import subprocess  # Lets Python run shell commands like kubectl and docker
from langchain_ollama import ChatOllama  # Connects Python to the local Ollama AI model
from langchain_core.tools import tool  # Lets us create tools the AI can call
from langchain.agents import create_agent  # Creates the AI agent that can use tools

# This creates the AI model connection using Ollama.
# It tells Python which local model to use and how much context it can handle.
llm = ChatOllama(
    model="llama3.2",  # The LLM model installed in Ollama
    temperature=0,  # A lower value makes answers more focused and predictable
    num_ctx=8192,  # Maximum context size
)

# This is a tool the agent can use to inspect Kubernetes pods.
# It runs the kubectl command in the terminal and returns the output.
@tool
def get_pods():
    """
    Lists the pods of a running Kubernetes cluster.
    """
    result = subprocess.run(["kubectl", "get", "pods", "-A"], capture_output=True, text=True)
    return result.stdout or result.stderr  # Return command output or any error message


# This is another tool the agent can use to inspect local Docker containers.
@tool
def get_docker_containers():
    """
    Lists the running Docker containers.
    """
    result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
    return result.stdout or result.stderr  # Return command output or any error message

@tool
def get_system_info():
    """Gets basic information about the current computer."""
    import platform
    import psutil

    return {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "CPU Usage": f"{psutil.cpu_percent()}%",
        "RAM Usage": f"{psutil.virtual_memory().percent}%",
        "Disk Usage": f"{psutil.disk_usage('/').percent}%"
    }

# This creates the actual agent.
# The agent combines the LLM with the available tools.


agent = create_agent(
    model=llm,
    tools=[get_pods, get_docker_containers, get_system_info],
    system_prompt=(
        "You are a DevOps assistant. Use the available tools to inspect live systems. "
        "Never invent results. Keep responses concise."
    )
)

question = input("Ask your Agent a Question: >")  # Ask the user for a question

response = agent.invoke({"messages": [("user", question)]})  # Send the question to the agent

print(response["messages"][-1].content)  # Print the final answer from the AI model






