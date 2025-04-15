from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
import datasets
from langchain.docstore.document import Document


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        super().__init__()
        self.docs = docs
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(self.docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([str(doc.page_content) for doc in results[:3]])
        else:
            return "No matching guest information found."


guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

docs = [
   Document(
       page_content="\n".join([
           f"Name: {guest['name']}",
           f"Relation: {guest['relation']}",
           f"Description: {guest['description']}",
           f"Email: {guest['email']}"
       ]),
       metadata={"name": guest["name"]}
   )
   for guest in guest_dataset
]
# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)