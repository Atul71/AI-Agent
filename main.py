from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


model = OllamaLLM(model="llama3.2")


prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content="You are Alice, a chatbot that helps humans get as much information as possible"
    ),#persistent system prompt
    MessagesPlaceholder(
        variable_name ="chat_history"
    ),
    HumanMessagePromptTemplate.from_template(
        "{human_input}"
    )
    
])

memory =  ConversationBufferMemory(
    memory_key = "chat_history",
    return_messages=True
)

chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory
)

while True:     
    print("\n\n-----------------------------")
    print("\n\n-----------------------------")
    question = input("Ask Alice a question (q to quit)")
    if question.lower()=='q':
        break
    
    result = chain.invoke({
        "human_input": question
    })

    print("\n Alice:", result['text'])
    
    

