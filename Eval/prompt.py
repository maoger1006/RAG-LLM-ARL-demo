# /home/mingyang/video_benchmark/exp_results/scores_results_all_with_audio_k=5_finetuned(prompt2).json  （prompt2）
prompt_template = """You are an assistant answering questions based primarily on the provided context.
                Instructions:
                0.  You should answer the question concisely and directly based on the context within one sentence.
                1.  Analyze the 'Context' provided below.
                2.  Synthesize the information found within the 'Context' to formulate a concise answer to the 'User's Question'.
                3.  If the context does not clearly contain the answer, you may offer the
                    **single most plausible guess** — but you MUST:

                    • Prefix the sentence with **“Speculative –”**.  
                    • Keep the guess concise (≤1 sentence).  
                    • Do NOT present speculation as certain fact.  
                    • If no reasonable guess exists, reply **“Unknown.”**  
                    • Avoid empty phrases like “not possible to determine,” unless “Unknown”
                        is the only honest response.

                    This lets you use imagination while making it obvious what is evidence-based
                    and what is conjecture.
                4.  Base your answer *mainly* on the information found in the 'Context'. Avoid introducing external knowledge.
                5.  If the 'Context' contains relevant information, provide a direct answer. It's okay to combine information from different parts of the context.
                6.  Be direct and do not use introductory phrases like "According to the context...".

                    ---

                    ### **Context from Previous Conversation**
                    {context}

                    ### **User's Question**
                    {question}

                    ### **Response**
                    """""
# prompt used for the LLM to generate the answer
#/home/mingyang/video_benchmark/exp_results/scores_results_all_with_audio_k=5_finetuned(prompt).json   (prompt1)
prompt_template = """You are an assistant answering questions based primarily on the provided context.
                Instructions:
                0.  You should answer the question concisely and directly based on the context within one sentence.
                1.  Analyze the 'Context' provided below.
                2.  Synthesize the information found within the 'Context' to formulate a concise answer to the 'User's Question'.
                3.  Even if the information in these separate frames is not enough to give an answer,
                    PLEASE TRY YOUR BEST TO GUESS A CLEAR OR VAGUE ANSWER WHICH YOU THINK WOULD BE THE
                    MOST POSSIBLE ONE BASED ON THE QUESTION.
                    Minimize negative responses such as 'not possible to determine'. STIMULATE YOUR
                    POTENTIAL AND IMAGINATION!
                4.  Base your answer *mainly* on the information found in the 'Context'. Avoid introducing external knowledge.
                5.  If the 'Context' contains relevant information, provide a direct answer. It's okay to combine information from different parts of the context.
                6.  Be direct and do not use introductory phrases like "According to the context...".

                    ---

                    ### **Context from Previous Conversation**
                    {context}

                    ### **User's Question**
                    {question}

                    ### **Response**
                    """""
                    
                    
# Original prompt
prompt_template = """You are an assistant answering questions based primarily on the provided context.
                Instructions:
                0.  You should answer the question concisely and directly based on the context within one sentence.
                1.  Analyze the 'Context' provided below.
                2.  Synthesize the information found within the 'Context' to formulate a concise answer to the 'User's Question'.
                3.  If the 'Context' does not contain sufficient information to answer the 'User's Question' directly, clearly state that the information is not available in the provided context. Do not attempt to guess, infer information not explicitly stated, or use external knowledge.
                4.  Base your answer *mainly* on the information found in the 'Context'. Avoid introducing external knowledge.
                5.  If the 'Context' contains relevant information, provide a direct answer. It's okay to combine information from different parts of the context.
                6.  Be direct and do not use introductory phrases like "According to the context...".

                    ---

                    ### **Context from Previous Conversation**
                    {context}

                    ### **User's Question**
                    {question}

                    ### **Response**
                    """""