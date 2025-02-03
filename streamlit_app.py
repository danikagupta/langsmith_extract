import streamlit as st
from langsmith import Client
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

def run_one_msg(llm_messages,time_taken):
  model = ChatGroq(model=st.secrets['GROQ_MODEL'], temperature=0, api_key=st.secrets['GROQ_API_KEY'])
  #llm_messages = [SystemMessage(content="Explain the meaning of life in 10 words or less.")]
  llm_response = model.invoke(llm_messages)
  print(f"{llm_response.content=}\n{llm_response.response_metadata['token_usage']['total_time']=} vs. {time_taken=}\n\n")
  return llm_response.content, llm_response.response_metadata['token_usage']['total_time']

def create_llm_message(original_list):
  new_list=[]
  for dict in original_list:
    if dict['type']=='system':
      new_list.append(SystemMessage(content=dict['content']))
    elif dict['type']=='human':
      new_list.append(HumanMessage(content=dict['content']))
    else:
      new_list.append(AIMessage(content=dict['content']))
  print(f"DEBUG: CREATE-LLM-MESSAGE: {new_list=}")
  return new_list

def run_groq(li):
    total_time_taken=0
    original_time_taken=0
    result_list=[]
    my_bar = st.progress(0, text=f"Groq will process {len(li)} calls")
    with st.sidebar.expander("Groq logs"):
        for i,l in enumerate(li):
            inp=l['in1']
            out1=l['out1']
            time1=l['time1']

            llm_messages=create_llm_message(l['in1'])
            resp,time_taken=run_one_msg(llm_messages,l['time1'])
            total_time_taken+=time_taken
            original_time_taken+=time1
            st.write(f"{resp=}, {time_taken=}")
            result_list.append({"inp":str(inp)[:50],"LLM time":time1,"Groq time":time_taken,"LLM Output":out1,"Groq Output":resp})

            pct_complete=int(100*(i+1)/len(li))
            my_bar.progress(pct_complete,text=f"Finished {i+1} of {len(li)} calls")

    savings=100*(original_time_taken-total_time_taken)/original_time_taken
    st.header(f"Saved {savings:.1f}%, Groq time: {total_time_taken:.1f} vs. LLM time: {original_time_taken:.1f} seconds") 
    return result_list   

def get_run_info(run_id):
    project_runs2a=client.list_runs(run_ids=[run_id],error=False)
    prlist2a=list(project_runs2a)
    print(f"Get_run_info: {prlist2a=}")
    one_run=prlist2a[0]
    print(f"Get_run_info: {one_run=}")
    child_runs=one_run.child_run_ids
    print(f"Get_run_info: {child_runs=}")
    if child_runs is None:
        child_runs=[run_id]
        print(f"Get_run_info UPDATED: {child_runs=}")
    project_runs2b=client.list_runs(run_ids=child_runs,error=False)
    prlist2b=list(project_runs2b)
    print(f"Get_run_info: {prlist2b=}")

    full_list=[]

    for run in prlist2b:
        if run.run_type=='llm':
            inputs=run.inputs
            msglist=[]
            for msg in inputs["messages"][0]:
                msglist.append(msg['kwargs'])
            outputs=run.outputs
            resp_output=outputs['generations'][0][0]['text']
            events=run.events
            starttime=0
            endtime=0
            for event in events:
                if event['name']=='start':
                    starttime=event['time']
                if event['name']=='end':
                    endtime=event['time']
            start_dt = datetime.fromisoformat(starttime)
            end_dt = datetime.fromisoformat(endtime)
            time_difference = end_dt - start_dt
            full_list.append({'in1':msglist,'out1':resp_output,'time1':time_difference.total_seconds()})
    return full_list

def show_run(pr):
    st.header(f"Run: {pr.id}")
    li=get_run_info(pr.id)
    with st.sidebar.expander("Full list"):
        for i,l in enumerate(li):
            st.write(f"Element {i}: {l}\n")
    if st.sidebar.button("Run Groq"):
        results=run_groq(li)
        st.dataframe(results)

def show_project(project_id,project_name):
    #st.header(f"F PROJECT: {project_name=}, {project_id=}")
    project_runs=client.list_runs(project_name=project_name,error=False, is_root=True, select=["inputs", "outputs", "start_time"])
    prlist=list(project_runs)
    idlist=[pr.id for pr in prlist]
    #tslist=[pr.start_time for pr in prlist]
    #outlist=[str(pr.outputs)[:30] for pr in prlist]
    tsoutlist=[str(pr.start_time)+" "+str(pr.outputs)[:30] for pr in prlist]
    pr_dict=dict(zip(tsoutlist,prlist))
    selection=st.sidebar.selectbox("Select run:",tsoutlist)
    if selection:
        pr=pr_dict[selection]
        #st.write(f"Selected project {id=} for {selection=}")
        show_run(pr)
    #st.write(f"{idlist=}")
    #st.write(f"{tslist=}")
    #st.write(f"{outlist=}")
    #st.write(f"{tsoutlist=}\n\n\n")
    #for i,pr in enumerate(prlist):
    #    id=pr.id
    #    inputs = pr.inputs if hasattr(pr, 'inputs') else {}
    #    outputs = pr.outputs if hasattr(pr, 'outputs') else {}
    #    timestamp = pr.start_time if hasattr(pr, 'start_time') else "N/A"
    #    st.write(f"Run {i}:{id=}\n\n{inputs=}\n\n{outputs=}\n\n{timestamp=}\n\n")

def one_run():
    if 'projects' not in st.session_state:
        projects = list(client.list_projects())
        st.session_state['projects']=projects
    else:
        projects=st.session_state['projects']

    project_names=[project.name for project in projects]
    project_ids=[project.id for project in projects]
    project_dict=dict(zip(project_names,project_ids))

    with st.sidebar.expander("All projects"):
        for project in projects:
            st.write(f"- {project.name} (ID: {project.id})")

    selection=st.sidebar.selectbox("Select project",project_names, index=None)
    if selection:
        selected_id=project_dict[selection]
        #st.write(f"You selected project: {selection=},{selected_id=}") 
        show_project(selected_id,selection) 
    


#
# Main
#

print("Starting main...")
st.set_page_config(layout="wide")

if 'lskey' in st.session_state:
    lskey=st.session_state['lskey']
    print(f"Fetched from cache {lskey=}")
else:
    lskey=st.sidebar.text_input("Langsmith API key")
    if lskey:
        st.session_state['lskey']=lskey
        print(f"Just accepted {lskey=}")

if lskey:   
    print("Starting up...") 
    client=Client(api_key=lskey,api_url="https://api.smith.langchain.com")
    one_run()