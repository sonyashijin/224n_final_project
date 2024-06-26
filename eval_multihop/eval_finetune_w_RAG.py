import pickle
from langchain_community.document_loaders import WikipediaLoader
from transformers import pipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import csv
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np

import logging
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
logging.info("This is an info message")

checkpoint_path = "~/final_project/peft-training/checkpoint-1250"
# Load documents from pickle file
with open('loaded_articles_2024_new.pkl', 'rb') as f:
    documents = pickle.load(f)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
device_map = {"": 0}
base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                      device_map='auto',
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=hf_token)

ft_model = PeftModel.from_pretrained(base_model, checkpoint_path,torch_dtype=torch.float16,is_trainable=False)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen(model, p, maxlen=100, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample, num_return_sequences=1, temperature=0.1, num_beams=1, top_p=0.95).to('cpu')
    return eval_tokenizer.batch_decode(res, skip_special_tokens=True)


# Flatten the list of lists into a single list of documents
documents_flattened = [doc for sublist in documents for doc in sublist]

# Initialize text splitter
chunk_size = 1000  # Adjust as needed
chunk_overlap = 100  # Optional overlap for better context
text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def split_document(document):
    chunks = text_splitter.split_text(document.page_content)
    return [Document(page_content=chunk) for chunk in chunks]

chunked_documents = []
for doc in documents_flattened:
    chunked_documents.extend(split_document(doc))

chunked_texts = [doc.page_content for doc in chunked_documents]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunked_texts, convert_to_tensor=True)
# np.save('embeddings.npy', embeddings)
print("embedding_done")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.cpu().numpy())

def retrieve_top_k(question, k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, k)
    return [chunked_texts[idx] for idx in indices[0]]

def baseline_rag(question):
    retrieved_documents = retrieve_top_k(question)
    context = " ".join(retrieved_documents)
    input_text = f"Answer the question with given context: {question}\nContext: {context}\nAnswer:"
    generated = gen(base_model, input_text, 100)
    # Print the full generated output for debugging
    print("Full generated output:", generated)

    # Get the generated text and remove the input_text part to isolate the answer
    full_generated_text = generated[0]
    answer = full_generated_text.replace(input_text, '').strip()

    # Further process to extract the answer part only, if needed
    answer_lines = answer.split('\n')
    if len(answer_lines) > 1:
        answer = answer_lines[0].strip()  # Typically, the first line after input_text is the answer

    # Fallback if answer is still incorrectly extracted
    if not answer or "Question" in answer:
        answer = "Answer not found or incorrect extraction"

    print("Extracted Answer:", answer)
    return answer, context


def calculate_accuracy(predefined_qa_pairs):
    correct_answers = 0
    answer_dict = {}
    for qa_pair in predefined_qa_pairs:
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"].strip().lower().split()  # Split into words and handle case
        expected_context = " ".join(qa_pair["chunks"])  # Combine chunks to form the expected context

        # Generate an answer using the baseline_rag function
        generated_answer, retrieved_context = baseline_rag(question)
        logging.info(f"Question: {question}")
        logging.info(f"Generated Answer: {generated_answer}")

        # Check if the generated answer matches the expected answer up to the first three words
        if (len(expected_answer) >= 1 and generated_answer[:1] == expected_answer[:1]) or \
           (len(expected_answer) >= 2 and generated_answer[:2] == expected_answer[:2]) or \
           (len(expected_answer) >= 3 and generated_answer[:3] == expected_answer[:3]):
            correct_answers += 1
        
        answer_dict[question] = {
            "expected": expected_answer, 
            "generated": generated_answer, 
            "retrieved_context": retrieved_context,
            "expected_context": expected_context
        }
        
    # Calculate accuracy
    accuracy = correct_answers / len(predefined_qa_pairs)

    return accuracy, answer_dict




if __name__ == "__main__":
    predefined_qa_pairs = [
        {
            "question": "How many years after the release of Taylor Swift's tenth studio album, Midnights, was 'The Tortured Poets Department' released?",
            "answer": "Approximately 1.5 years",
            "chunks": [
                "Swift released her tenth studio album, Midnights, on October 21, 2022, to critical praise and commercial success.",
                "The Tortured Poets Department was released on April 19, 2024."
            ]
        },
        {
            "question": "What was the collective sales figure of 'The Tortured Poets Department' in its first week in the United States, including both album-equivalent units and pure sales?",
            "answer": "2.61 million units",
            "chunks": [
                "The album broke various sales and streaming records.",
                "In the United States, The Tortured Poets Department debuted atop the Billboard 200 with first-week 2.6 million album-equivalent units, including 1.9 million pure sales."
            ]
        },
        {
            "question": "Which two cities hosted the studios where Taylor Swift recorded 'The Tortured Poets Department' and its subsequent double album, 'The Anthology'?",
            "answer": "New York City and Los Angeles",
            "chunks": [
                "Swift began writing The Tortured Poets Department shortly after finishing her tenth studio album, Midnights (2022), and continued developing it during the Eras Tour in 2023.",
                "The album was recorded at various studios, including Audu (Brooklyn), Big Mercy (New York City), Conway Recording (Hollywood), Electric Lady (New York City), and Electric Feel (Los Angeles)."
            ]
        },
        {
            "question": "How did Taylor Swift announce the release of 'The Tortured Poets Department' during the 66th Annual Grammy Awards?",
            "answer": "She announced it during her acceptance speech for Best Pop Vocal Album for 'Midnights'",
            "chunks": [
                "At the 66th Annual Grammy Awards on February 4, 2024, Swift won Best Pop Vocal Album and Album of the Year for Midnights.",
                "During her acceptance speech for the former category, she announced The Tortured Poets Department as a new original studio album that she had worked on since 2022."
            ]
        },
        {
            "question": "What records did 'The Tortured Poets Department' break on Spotify within its first week of release?",
            "answer": "It achieved the highest single-day and single-week global streams for an album, with over 1.76 billion streams globally within its first week and surpassing 200 million and then 300 million streams in a single day",
            "chunks": [
                "The album broke various sales and streaming records.",
                "On Spotify, it became the album with the most number of pre-saves; the most streamed album in a single day by surpassing 200 million and then 300 million streams, breaking the all-time record previously held by Swift's Midnights.",
                "It amassed 1.76 billion streams globally within its first week of availability, registering an all-time record."
            ]
        },
        {
            "question": "Name two specific locations where Taylor Swift promoted the album through QR code murals.",
            "answer": "Various cities worldwide",
            "chunks": [
                "The album was promoted on digital platforms like Apple Music, Spotify, YouTube, Instagram, and Threads, prompting Swifties to search for Easter eggs.",
                "QR code murals in various cities worldwide that lead to unlisted YouTube shorts on Swift's channel."
            ]
        },
        {
            "question": "Which bonus tracks correspond to the four physical editions of 'The Tortured Poets Department'?",
            "answer": "'The Manuscript,' 'The Bolter,' 'The Albatross,' and 'The Black Dog'",
            "chunks": [
                "Swift announced four physical editions that were each titled after a corresponding bonus track: 'The Manuscript', 'The Bolter', 'The Albatross', and 'The Black Dog'."
            ]
        },
        {
            "question": "How did critics describe the musical and lyrical style of the second part of 'The Tortured Poets Department'?",
            "answer": "Critics likened it to Swift's 2020 albums 'Folklore' and 'Evermore,' with an acoustic, folk-oriented sound instrumented by picked acoustic guitar, soft piano, and subtle synths",
            "chunks": [
                "The second part of the double album, subtitled The Anthology, mostly consists of chamber pop and folk-pop piano ballads.",
                "Swift and Dessner produced the majority of the second volume, which has an acoustic, folk-oriented sound instrumented by picked acoustic guitar, soft piano, and subtle synths, which critics likened to the sound of Swift's 2020 albums Folklore and Evermore."
            ]
        },
        {
            "question": "Which artist featured on the lead single 'Fortnight' from 'The Tortured Poets Department'?",
            "answer": "Post Malone",
            "chunks": [
                "The Tortured Poets Department consists of 16 standard songs and features two guest acts—the American rapper Post Malone on the lead single 'Fortnight' and the English indie rock band Florence and the Machine, led by the singer-songwriter Florence Welch, on the song 'Florida!!!'."
            ]
        },
        {
            "question": "How many songs from 'The Anthology' debuted on the Billboard Hot 100, and what record did Taylor Swift set with this achievement?",
            "answer": "All 31 songs from 'The Anthology' debuted on the Billboard Hot 100, and Taylor Swift became the first artist to monopolize the first 14 positions of the Billboard Hot 100 simultaneously",
            "chunks": [
                "All 31 songs from The Anthology debuted on the Billboard Hot 100, occupying the entire top 14 simultaneously for the first time in chart history.",
                "Swift set the record for most simultaneous entries by a female artist (32) and became the first woman to surpass 50 career top-10 songs."
            ]
        }
    ]

    predefined_qa_pairs.extend([
        {
            "question": "What inspired Greta Gerwig to create a film that would both celebrate and critique Barbie, and how did she convey these themes in the movie?",
            "answer": "Gerwig was inspired by her childhood experiences with Barbie and societal pressures on American teenage girls. She conveyed these themes by juxtaposing Barbie's celebration of feminism with the controversial beauty standards and social expectations.",
            "chunks": [
                "Gerwig was inspired by her childhood experiences with Barbie, which included both fond memories and the complex societal pressures on American teenage girls.",
                "She wanted to create a film that would both celebrate and critique Barbie, addressing themes of feminism, beauty standards, and social expectations.",
                "In the movie, these themes are conveyed through a narrative that juxtaposes Barbie's positive impact on women's empowerment with the controversial aspects of her portrayal."
            ]
        },
        {
            "question": "How did Greta Gerwig's personal experiences and societal influences shape the narrative of the Barbie film?",
            "answer": "Gerwig's personal experiences with Barbie and societal influences on American teenage girls shaped the narrative by addressing themes of feminism and social expectations.",
            "chunks": [
                "Gerwig's personal experiences with Barbie included both fond memories and the complex societal pressures on American teenage girls.",
                "These personal experiences, along with broader societal influences, shaped the film's narrative to address themes of feminism, beauty standards, and social expectations."
            ]
        },
        {
            "question": "What themes did Greta Gerwig explore in the Barbie film, and how were they represented in the storyline?",
            "answer": "Gerwig explored themes of feminism, beauty standards, and social expectations. These were represented by juxtaposing Barbie's celebration of women's empowerment with the controversial aspects of her portrayal.",
            "chunks": [
                "The film addresses themes of feminism, beauty standards, and social expectations.",
                "These themes are represented through a storyline that juxtaposes Barbie's positive impact on women's empowerment with the controversial aspects of her portrayal."
            ]
        },
        {
            "question": "What parallels did Greta Gerwig draw between Barbie's journey in the film and classic literary works?",
            "answer": "Gerwig drew parallels between Barbie's journey and the story of Adam and Eve, as well as being inspired by John Milton's Paradise Lost.",
            "chunks": [
                "Gerwig began her writing by interpreting Barbie as living in a utopia and eventually experiencing reality, drawing parallels to the story of Adam and Eve.",
                "She took inspiration from John Milton's Paradise Lost, particularly being inspired by the concept that there is 'no poetry without pain.'"
            ]
        },
        {
            "question": "How did the film's narrative reflect the themes of societal expectations and personal growth?",
            "answer": "The film's narrative reflected these themes by showing Barbie's transition from a utopian world to reality, where she confronts societal expectations and personal growth.",
            "chunks": [
                "The film's narrative begins with Barbie living in a utopia and eventually experiencing reality, where she confronts societal expectations.",
                "Barbie's journey from childhood to adolescence reflects personal growth, though the film ultimately ends up about being human."
            ]
        },
        {
            "question": "What role did humor play in the critical reception of the Barbie film?",
            "answer": "Humor played a significant role in the critical reception, with critics praising its use to address past criticisms of the Barbie brand while also infusing subversive storytelling.",
            "chunks": [
                "Critics praised the script for addressing past criticisms of the Barbie brand's portrayal of women and lack of diversity while infusing humor.",
                "The website's consensus reads: 'Barbie is a visually dazzling comedy whose meta humor is smartly complemented by subversive storytelling.'"
            ]
        },
        {
            "question": "How did Greta Gerwig's approach to directing Barbie reflect her vision for the film?",
            "answer": "Gerwig's approach to directing reflected her vision by embracing maximalism and heightened theatricality, which allowed her to deal with big ideas amidst anarchic play.",
            "chunks": [
                "Gerwig grounded the film in what she described as a 'heightened theatricality that allows you to deal with big ideas in the midst of anarchic play.'",
                "She also described the film as being anarchic, unhinged, and humanist."
            ]
        },
        {
            "question": "In what ways did the film's production design contribute to its overall themes?",
            "answer": "The production design contributed to the film's themes by using authentic artificiality and tangible artifice to enhance the storytelling.",
            "chunks": [
                "Gerwig was inspired by classic Technicolor musicals and aimed for a high level of 'authentic artificiality.'",
                "The production design used painted backdrops and tangible artifice to enhance the storytelling."
            ]
        },
        {
            "question": "How did the characters of Barbie and Ken evolve throughout the film?",
            "answer": "Barbie evolved by confronting societal expectations and finding her own identity, while Ken learned about patriarchy and sought respect and autonomy.",
            "chunks": [
                "Barbie decides to become human and return to the real world, finding her own identity.",
                "Ken learns about patriarchy and feels respected for the first time, seeking an autonomous identity."
            ]
        },
        {
            "question": "What messages did Greta Gerwig aim to convey through Barbie's journey and experiences in the real world?",
            "answer": "Gerwig aimed to convey messages about the importance of self-discovery, confronting societal expectations, and the impossibility of perfection.",
            "chunks": [
                "Barbie's journey involves confronting all the things that were shielded from her in Barbieland, reflecting self-discovery.",
                "Gerwig wanted to convey the impossibility of perfection and the importance of self-worth."
            ]
        },
        {
            "question": "How did the film critique consumerism and beauty standards associated with Barbie?",
            "answer": "The film critiqued consumerism and beauty standards by juxtaposing Barbie's celebration of feminism with the controversial aspects of her portrayal.",
            "chunks": [
                "The film deliberately juxtaposed contradictory messaging, such as critiquing consumerism yet glamorizing plastic products.",
                "Barbie's journey includes being criticized for encouraging unrealistic beauty standards."
            ]
        },
        {
            "question": "What role did societal pressures play in the development of the film's narrative and themes?",
            "answer": "Societal pressures played a significant role by influencing the narrative to address themes of feminism, beauty standards, and the challenges faced by women.",
            "chunks": [
                "Gerwig's personal experiences with societal pressures influenced the narrative to address themes of feminism and beauty standards.",
                "The film reflects societal pressures on American teenage girls and their impact on women's empowerment."
            ]
        },
        {
            "question": "How did the film's ending reinforce its central themes and messages?",
            "answer": "The film's ending reinforced its themes by showing Barbie deciding to become human and emphasizing the importance of self-worth and autonomy.",
            "chunks": [
                "The ending features Barbie deciding to become human, reinforcing the theme of self-worth and autonomy.",
                "Gerwig described the ending as a 'mic drop kind of joke,' instilling confidence in younger girls."
            ]
        },
        {
            "question": "What impact did the film's release have on public perception of the Barbie brand?",
            "answer": "The film's release had a significant impact by addressing past criticisms of the Barbie brand and promoting a more inclusive and empowering image.",
            "chunks": [
                "Critics praised the film for addressing past criticisms of the Barbie brand's portrayal of women and lack of diversity.",
                "The film's narrative and themes promoted a more inclusive and empowering image of Barbie."
            ]
        },
        {
            "question": "How did the film explore the concept of gender roles and expectations?",
            "answer": "The film explored gender roles and expectations by showing the contrast between Barbieland's matriarchy and the real world's patriarchy.",
            "chunks": [
                "The Kens indoctrinate the Barbies into submissive roles, reflecting gender expectations in the real world.",
                "The film explores the negative consequences of hierarchical power structures and the contrast between matriarchy and patriarchy."
            ]
        },
        {
            "question": "What were the main influences on Greta Gerwig's screenplay for the Barbie film?",
            "answer": "The main influences included Gerwig's childhood experiences, societal pressures, and classic literary works like Paradise Lost.",
            "chunks": [
                "Gerwig was influenced by her childhood experiences with Barbie and societal pressures.",
                "She drew parallels to classic literary works like John Milton's Paradise Lost."
            ]
        },
        {
            "question": "How did the film address the idea of personal identity and self-worth?",
            "answer": "The film addressed personal identity and self-worth by showing Barbie's journey of self-discovery and confronting societal expectations.",
            "chunks": [
                "Barbie's journey involves finding her own identity and confronting societal expectations.",
                "The film emphasizes the importance of self-worth and the idea that 'just being yourself is enough.'"
            ]
        },
        {
            "question": "What role did nostalgia play in the audience's reception of the film?",
            "answer": "Nostalgia played a significant role by drawing on the audience's fond memories of Barbie while also challenging their perceptions.",
            "chunks": [
                "The film's use of humor and subversive storytelling resonated with audiences' nostalgic memories of Barbie.",
                "Critics noted that the film is grounded largely in audience nostalgia."
            ]
        },
            {
            "question": "What were the main challenges faced by the production team of Oppenheimer, and how did they overcome them?",
            "answer": "The production team faced challenges with practical effects, filming in IMAX, and maintaining historical accuracy, which they overcame through meticulous planning and extensive research.",
            "chunks": [
                "Nolan used extensive practical effects, with minimal compositing, to create an immersive experience.",
                "The production team conducted extensive research to ensure historical accuracy, including using IMAX 65 mm and 65 mm large-format film."
            ]
        },
        {
            "question": "How did Oppenheimer's internal conflict and the political repercussions he faced shape the film's narrative?",
            "answer": "Oppenheimer's internal conflict and the political repercussions he faced were central to the narrative, emphasizing his guilt and the questioning of his loyalty to the United States.",
            "chunks": [
                "Oppenheimer is guilt-ridden and haunted by the destruction and mass fatalities.",
                "His loyalty to the United States is questioned, and his past communist ties are exploited."
            ]
        },
        {
            "question": "What thematic parallels can be drawn between Barbie's and Oppenheimer's struggles with societal expectations?",
            "answer": "Both Barbie and Oppenheimer struggle with societal expectations, with Barbie confronting beauty standards and feminism, while Oppenheimer faces moral responsibility and political scrutiny.",
            "chunks": [
                "Barbie's journey involves confronting societal expectations and finding her own identity, reflecting themes of feminism and beauty standards.",
                "Oppenheimer's narrative explores his internal conflict and the political repercussions of his actions, emphasizing themes of guilt and responsibility."
            ]
        },
        {
            "question": "How did Greta Gerwig and Christopher Nolan use visual aesthetics to enhance the storytelling in Barbie and Oppenheimer respectively?",
            "answer": "Gerwig used 'authentic artificiality' and painted backdrops, while Nolan used practical effects and IMAX film to create immersive experiences.",
            "chunks": [
                "Gerwig was inspired by classic Technicolor musicals and aimed for a high level of 'authentic artificiality,' using painted backdrops and tangible artifice.",
                "Nolan used extensive practical effects and a combination of IMAX 65 mm and 65 mm large-format film, including black-and-white photography, to enhance the storytelling."
            ]
        },
        {
            "question": "What were the unique promotional strategies used for both Barbie and Oppenheimer to generate audience interest?",
            "answer": "Barbie used QR code murals and social media engagement, while Oppenheimer utilized the 'Barbenheimer' phenomenon and Nolan's reputation for high-quality films.",
            "chunks": [
                "The Barbie film included QR code murals in various cities worldwide that lead to unlisted YouTube shorts on Swift's channel and special shimmer effects on Threads posts.",
                "Oppenheimer benefited from the 'Barbenheimer' phenomenon, which boosted ticket sales for both films and encouraged audiences to see them as a double feature."
            ]
        },
        {
            "question": "How did the narrative structures of Barbie and Oppenheimer differ in their approach to exploring complex themes?",
            "answer": "Barbie used a juxtaposition of utopian and real-world experiences to explore feminism and beauty standards, while Oppenheimer alternated between subjective and objective perspectives to examine guilt and responsibility.",
            "chunks": [
                "The Barbie film's narrative begins with Barbie living in a utopia and eventually experiencing reality, where she confronts societal expectations.",
                "Oppenheimer alternated between scenes in color and black-and-white to convey the story from both subjective and objective perspectives, exploring complex themes of guilt and responsibility."
            ]
        },
        {
            "question": "What were the critical responses to the portrayal of historical accuracy in Oppenheimer, and how did it impact the film's reception?",
            "answer": "Critics praised the historical accuracy of Oppenheimer, which enhanced the film's credibility and reception.",
            "chunks": [
                "The production team conducted extensive research to ensure historical accuracy, including using IMAX 65 mm and 65 mm large-format film.",
                "Critics noted the film's nuanced portrayal of historical events and its commitment to accuracy, which positively impacted its reception."
            ]
        },
        {
            "question": "How did the film's narrative and visual style reflect Greta Gerwig's and Christopher Nolan's directorial visions in Barbie and Oppenheimer?",
            "answer": "Gerwig's narrative and visual style in Barbie used heightened theatricality and 'authentic artificiality,' while Nolan's style in Oppenheimer employed practical effects and IMAX film to reflect their respective visions.",
            "chunks": [
                "Gerwig grounded the Barbie film in a 'heightened theatricality that allows you to deal with big ideas in the midst of anarchic play,' using 'authentic artificiality.'",
                "Nolan used extensive practical effects and a combination of IMAX 65 mm and 65 mm large-format film to create an immersive experience in Oppenheimer."
            ]
        },
        {
            "question": "What messages did the Barbie film convey about the importance of self-worth and autonomy, and how were these themes portrayed?",
            "answer": "The Barbie film conveyed the importance of self-worth and autonomy by showing Barbie's journey of self-discovery and decision to become human.",
            "chunks": [
                "The ending features Barbie deciding to become human, reinforcing the theme of self-worth and autonomy.",
                "Barbie's journey involves finding her own identity and confronting societal expectations, emphasizing that 'just being yourself is enough.'"
            ]
        },
        {
            "question": "How did Oppenheimer's personal struggles and political repercussions shape the film's exploration of guilt and responsibility?",
            "answer": "Oppenheimer's personal struggles and political repercussions highlighted his guilt and responsibility, emphasizing the moral complexities of his actions.",
            "chunks": [
                "Oppenheimer is guilt-ridden and haunted by the destruction and mass fatalities.",
                "His loyalty to the United States is questioned, and his past communist ties are exploited."
            ]
        },
        {
            "question": "What were the main challenges faced by Greta Gerwig and Christopher Nolan in the production of Barbie and Oppenheimer?",
            "answer": "Gerwig faced challenges in achieving 'authentic artificiality,' while Nolan dealt with practical effects and IMAX filming to maintain historical accuracy.",
            "chunks": [
                "Gerwig aimed for a high level of 'authentic artificiality,' using painted backdrops and tangible artifice to enhance the storytelling.",
                "Nolan used extensive practical effects and a combination of IMAX 65 mm and 65 mm large-format film to maintain historical accuracy in Oppenheimer."
            ]
        },
        {
            "question": "How did the casting choices for Barbie and Oppenheimer reflect the directors' visions and the characters' complexities?",
            "answer": "Gerwig's casting included diverse and empowering choices, while Nolan's casting involved actors taking pay cuts and extensive preparation to reflect their characters' complexities.",
            "chunks": [
                "Gerwig's casting choices included diverse actors to reflect a more inclusive and empowering image of Barbie.",
                "Nolan's casting choices, including Cillian Murphy and Robert Downey Jr., involved actors taking pay cuts and engaging in extensive research and physical preparation."
            ]
        },
        {
            "question": "What themes of societal expectations and personal growth were explored in both Barbie and Oppenheimer, and how were they portrayed?",
            "answer": "Barbie explored societal expectations and personal growth through Barbie's journey of self-discovery, while Oppenheimer examined guilt and responsibility through Oppenheimer's internal conflict and political repercussions.",
            "chunks": [
                "Barbie's journey involves confronting societal expectations and finding her own identity, reflecting themes of feminism and beauty standards.",
                "Oppenheimer's narrative explores his internal conflict and the political repercussions of his actions, emphasizing themes of guilt and responsibility."
            ]
        },
        {
            "question": "How did the film's narrative structures in Barbie and Oppenheimer contribute to their exploration of complex themes and character development?",
            "answer": "Barbie used a juxtaposition of utopian and real-world experiences to explore feminism and beauty standards, while Oppenheimer alternated between subjective and objective perspectives to examine guilt and responsibility.",
            "chunks": [
                "The Barbie film's narrative begins with Barbie living in a utopia and eventually experiencing reality, where she confronts societal expectations.",
                "Oppenheimer alternated between scenes in color and black-and-white to convey the story from both subjective and objective perspectives, exploring complex themes of guilt and responsibility."
            ]
        }
    ])

    predefined_qa_pairs.extend([
        {
            "question": "What role did humor play in the critical reception of the Barbie film and how was it used to address past criticisms of the brand?",
            "answer": "Humor played a significant role, praised for addressing past criticisms while infusing subversive storytelling.",
            "chunks": [
                "Critics praised the script for addressing past criticisms of the Barbie brand's portrayal of women and lack of diversity while infusing humor.",
                "The website's consensus reads: 'Barbie is a visually dazzling comedy whose meta humor is smartly complemented by subversive storytelling.'"
            ]
        },
        {
            "question": "How did the production design of the Barbie film contribute to its overall themes and how did Gerwig ensure authenticity?",
            "answer": "The production design used 'authentic artificiality' and tangible artifice, inspired by classic Technicolor musicals.",
            "chunks": [
                "Gerwig was inspired by classic Technicolor musicals and aimed for a high level of 'authentic artificiality.'",
                "The production design used painted backdrops and tangible artifice to enhance the storytelling."
            ]
        },
        {
            "question": "How did the characters of Barbie and Ken evolve throughout the film and what messages did their journeys convey?",
            "answer": "Barbie found her own identity, while Ken learned about patriarchy, both reflecting self-worth and autonomy.",
            "chunks": [
                "Barbie decides to become human and return to the real world, finding her own identity.",
                "Ken learns about patriarchy and feels respected for the first time, seeking an autonomous identity."
            ]
        },
        {
            "question": "How did the themes of societal expectations and personal growth manifest in the Barbie film's narrative?",
            "answer": "The narrative showed Barbie's transition from utopia to reality, confronting societal expectations and personal growth.",
            "chunks": [
                "The film's narrative begins with Barbie living in a utopia and eventually experiencing reality, where she confronts societal expectations.",
                "Barbie's journey from childhood to adolescence reflects personal growth, though the film ultimately ends up about being human."
            ]
        },
        {
            "question": "What was the impact of the 'Barbenheimer' phenomenon on the box office performance of both Barbie and Oppenheimer?",
            "answer": "The 'Barbenheimer' phenomenon boosted ticket sales for both films, leading to significant box office success.",
            "chunks": [
                "Oppenheimer grossed over $976 million worldwide, becoming the third-highest-grossing film of 2023.",
                "The concurrent release of Barbie and Oppenheimer encouraged audiences to see both films as a double feature."
            ]
        },
        {
            "question": "In what ways did the Barbie film address consumerism and beauty standards?",
            "answer": "The film critiqued consumerism and beauty standards by juxtaposing Barbie's celebration of feminism with her controversial portrayal.",
            "chunks": [
                "The film deliberately juxtaposed contradictory messaging, such as critiquing consumerism yet glamorizing plastic products.",
                "Barbie's journey includes being criticized for encouraging unrealistic beauty standards."
            ]
        },
        {
            "question": "How did the film explore the concept of gender roles and expectations, and what contrasts were drawn between Barbieland and the real world?",
            "answer": "The film explored gender roles by contrasting Barbieland's matriarchy with the real world's patriarchy.",
            "chunks": [
                "The Kens indoctrinate the Barbies into submissive roles, reflecting gender expectations in the real world.",
                "The film explores the negative consequences of hierarchical power structures and the contrast between matriarchy and patriarchy."
            ]
        },
        {
            "question": "What were the primary influences on Greta Gerwig's screenplay for Barbie, and how did her personal experiences shape the film?",
            "answer": "Gerwig was influenced by her childhood experiences with Barbie and societal pressures, shaping the film's narrative.",
            "chunks": [
                "Gerwig's personal experiences with Barbie included both fond memories and the complex societal pressures on American teenage girls.",
                "These personal experiences, along with broader societal influences, shaped the film's narrative to address themes of feminism, beauty standards, and social expectations."
            ]
        },
        {
            "question": "How did the Barbie film's ending reinforce its central themes and messages?",
            "answer": "The ending showed Barbie deciding to become human, emphasizing self-worth and autonomy.",
            "chunks": [
                "The ending features Barbie deciding to become human, reinforcing the theme of self-worth and autonomy.",
                "Gerwig described the ending as a 'mic drop kind of joke,' instilling confidence in younger girls."
            ]
        },
        {
            "question": "What role did societal pressures play in the development of Barbie's narrative and themes?",
            "answer": "Societal pressures influenced the narrative to address themes of feminism, beauty standards, and challenges faced by women.",
            "chunks": [
                "Gerwig's personal experiences with societal pressures influenced the narrative to address themes of feminism and beauty standards.",
                "The film reflects societal pressures on American teenage girls and their impact on women's empowerment."
            ]
        },
        {
            "question": "How did Oppenheimer's development and filming process reflect Nolan's approach to historical biopics?",
            "answer": "Nolan's approach involved extensive practical effects and a focus on the subjective experience of Oppenheimer.",
            "chunks": [
                "Nolan used extensive practical effects, with minimal compositing.",
                "The film used a combination of IMAX 65 mm and 65 mm large-format film, including, for the first time, scenes in IMAX black-and-white film photography."
            ]
        },
        {
            "question": "How did the film Oppenheimer portray the long-term consequences of Oppenheimer's actions and his personal struggles?",
            "answer": "The film portrayed Oppenheimer's guilt and the political repercussions he faced, emphasizing his internal conflict.",
            "chunks": [
                "Oppenheimer is guilt-ridden and haunted by the destruction and mass fatalities.",
                "His loyalty to the United States is questioned, and his past communist ties are exploited."
            ]
        },
        {
            "question": "What themes did Christopher Nolan explore in Oppenheimer, and how did he convey these through the film's narrative and cinematography?",
            "answer": "Nolan explored themes of guilt, responsibility, and the impact of scientific advancements, using practical effects and a combination of color and black-and-white cinematography.",
            "chunks": [
                "Nolan wished to explore the phenomenon of delayed reactions, as he felt people are not 'necessarily confronted with the strongest or worst elements of [their actions] in the moment.'",
                "He chose to alternate between scenes in color and black-and-white to convey the story from both subjective and objective perspectives."
            ]
        },
        {
            "question": "How did the casting choices for Oppenheimer reflect Nolan's vision for the film, and what unique preparation did actors undertake?",
            "answer": "Nolan's casting choices, including Cillian Murphy and Robert Downey Jr., reflected his vision, with actors taking pay cuts and engaging in extensive research and physical preparation.",
            "chunks": [
                "Murphy lost an undisclosed amount of weight for the role in order to better match the real-life Oppenheimer's gaunt appearance.",
                "Robert Downey Jr., Matt Damon, and Emily Blunt took pay cuts to work on the film, with each earning $4 million in lieu of their usual $10–20 million upfront salary."
            ]
        },
        {
            "question": "What impact did the concurrent release of Barbie and Oppenheimer have on the cultural phenomenon known as 'Barbenheimer'?",
            "answer": "The concurrent release led to the 'Barbenheimer' phenomenon, boosting ticket sales for both films and encouraging audiences to see them as a double feature.",
            "chunks": [
                "Oppenheimer grossed over $976 million worldwide, becoming the third-highest-grossing film of 2023.",
                "The concurrent release of Barbie and Oppenheimer encouraged audiences to see both films as a double feature."
            ]
        },
        {
            "question": "How did Nolan's personal experiences and historical events influence his approach to the Oppenheimer screenplay?",
            "answer": "Nolan was influenced by his fears of nuclear holocaust and the resurgence of nuclear anxiety during the 2022 Russian invasion of Ukraine.",
            "chunks": [
                "Nolan had long desired to make a film about Oppenheimer, even prior to reading American Prometheus.",
                "He was also inspired by his fears of nuclear holocaust throughout childhood, as he lived during the era of Campaign for Nuclear Disarmament."
            ]
        },
        {
            "question": "How did the film's production design and cinematography enhance the portrayal of Oppenheimer's story?",
            "answer": "The production design and cinematography used practical effects and IMAX film to create an immersive experience, highlighting the historical and emotional aspects of Oppenheimer's story.",
            "chunks": [
                "The production design used a combination of IMAX 65 mm and 65 mm large-format film, including, for the first time, scenes in IMAX black-and-white film photography.",
                "Nolan used extensive practical effects, with minimal compositing."
            ]
        }])


    accuracy, answer_dict = calculate_accuracy(predefined_qa_pairs)
    print(f"Accuracy: {accuracy}")
    print(answer_dict)
    csv_filename = 'finetuneRAGresults.csv'

# Open the file in write mode
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'Expected Answer', 'Generated Answer', 'Retrieved Context', 'Expected Context']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for question, answers in answer_dict.items():
            row = {
                'Question': question,
                'Expected Answer': ' '.join(answers['expected']),
                'Generated Answer': answers['generated'],
                'Retrieved Context': answers['retrieved_context'],
                'Expected Context': answers['expected_context']
            }
            writer.writerow(row)

    print(f"Results saved to {csv_filename}")
