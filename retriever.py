
from vectorbase import hospital_vectorstore,doctor_vectorstore,route_vectorstore,profile_vectorstore
hospital_retriever=hospital_vectorstore.as_retriever(search_kwargs={"k":3})
doctor_retriever=doctor_vectorstore.as_retriever(search_kwargs={"k":3})
route_retriever=route_vectorstore.as_retriever(search_kwargs={"k":3})
profile_retriever=profile_vectorstore.as_retriever(search_kwargs={"k":3})
