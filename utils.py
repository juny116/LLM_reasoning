from langchain.schema import AIMessage, HumanMessage, SystemMessage

def messages_to_string(messages):
    text = ""
    for message in messages:
        text += message.content + "\n"
    return text



def messages_to_dict(messages):
    new_list = []
    for message in messages:
        if isinstance(message, HumanMessage):
            new_list.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            new_list.append({"role": "assistant", "content": message.content})
        elif isinstance(message, SystemMessage):
            new_list.append({"role": "system", "content": message.content})
    return new_list
