from actian_vectorai import VectorAIClient, DistanceMetric, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from collections import Counter

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = VectorAIClient("localhost:50051")
client.connect()

COLECAO = "conhecimento_clinico"

if client.collections.exists(COLECAO):
    client.collections.delete(COLECAO)

client.collections.create(
    COLECAO,
    vectors_config=VectorParams(size=384, distance=DistanceMetric.COSINE)
)

condicoes = [
    # ── Original 25 conditions ──────────────────────────────────────────────
    {
        "nome": "Classic dengue fever",
        "descricao_clinica": "Sudden high fever, strong headache, pain behind the eyes, body and joint aches, tiredness, may develop red skin rash",
        "perguntas_chave": [
            "How many days have you had the fever?",
            "Do you have pain behind your eyes or a severe headache?",
            "Have you noticed any red spots or rash on your skin?",
            "Are you feeling nauseous or vomiting?"
        ],
        "conduta_sugerida": "Advise rest and plenty of fluids. Refer to health center for blood test. Do not use aspirin or ibuprofen. Monitor for warning signs.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Hemorrhagic dengue fever",
        "descricao_clinica": "High fever that seems to get better then suddenly gets much worse, bleeding from nose, gums or skin, stomach pain, frequent vomiting, person becomes very weak and sleepy",
        "perguntas_chave": [
            "Did you have a fever that improved and then got suddenly worse?",
            "Is there any bleeding from your body — nose, gums or skin?",
            "Do you have severe abdominal pain?",
            "Are you very weak or having difficulty staying awake?"
        ],
        "conduta_sugerida": "Refer IMMEDIATELY to emergency. Life-threatening condition. Do not give any medications without medical guidance.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Malaria",
        "descricao_clinica": "Fever that comes and goes in cycles with severe chills, shaking, heavy sweating, headache and exhaustion. Person lives near or recently visited forested or riverside areas",
        "perguntas_chave": [
            "Does the fever come and go at regular times or in cycles?",
            "Do you have chills and shaking together with the fever?",
            "Have you been in a forested area, near a river, or traveled to another region recently?",
            "Have you had malaria before?"
        ],
        "conduta_sugerida": "Refer urgently to health center or unit with rapid malaria test. Do not treat without laboratory confirmation.",
        "urgencia": 2,
        "regioes": ["AM", "PA", "MT", "RO", "AC", "RR", "AP", "TO", "MA"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Leptospirosis",
        "descricao_clinica": "High fever, headache, body aches especially in the calves, red eyes. Person had contact with floodwater, sewage or mud",
        "perguntas_chave": [
            "Did you have contact with floodwater, sewage or mud?",
            "Do you have severe pain in your calves (back of lower legs)?",
            "Are your eyes red or yellowish?",
            "Are you producing less urine than usual, or is your urine dark-colored?"
        ],
        "conduta_sugerida": "Refer urgently to health center or emergency. Leptospirosis can progress quickly. Inform the doctor about contact with contaminated water.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Zika virus",
        "descricao_clinica": "Low-grade fever, itchy red rash all over the body, red eyes, joint pain, swelling in hands and feet. Urgent if the person is pregnant",
        "perguntas_chave": [
            "Do you have an itchy red rash on your body?",
            "Are your eyes red and itchy?",
            "Are you pregnant or could you be pregnant?",
            "Do you have pain or swelling in your joints?"
        ],
        "conduta_sugerida": "If pregnant: refer urgently for prenatal care. Otherwise: refer to health center. Advise on prevention of Aedes aegypti mosquito.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Chikungunya fever",
        "descricao_clinica": "Sudden high fever and very severe joint pain that may prevent movement, swollen joints, sometimes skin rash",
        "perguntas_chave": [
            "Is the joint pain so severe it prevents you from moving?",
            "Did the fever start suddenly?",
            "Do you have swelling in your hands, feet or knees?",
            "Are other people in your house or neighborhood having the same symptoms?"
        ],
        "conduta_sugerida": "Refer to health center. Advise rest, hydration and no aspirin. Joint pain may last weeks or months.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Tuberculosis",
        "descricao_clinica": "Cough that has lasted more than 3 weeks, sometimes with blood, night sweats, low-grade fever in the late afternoon, unexplained weight loss and constant fatigue",
        "perguntas_chave": [
            "How many weeks have you had this cough?",
            "Have you coughed up blood or bloody phlegm?",
            "Are you losing weight without dieting?",
            "Does anyone at home have tuberculosis or a prolonged cough?"
        ],
        "conduta_sugerida": "Refer to health center for sputum collection and chest X-ray. Advise covering mouth when coughing. Investigate household contacts.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Pneumonia",
        "descricao_clinica": "Fever, cough with yellow or green phlegm, difficulty breathing, chest pain when breathing deeply, person looks very tired and short of breath",
        "perguntas_chave": [
            "Are you having difficulty breathing or feeling short of breath?",
            "Do you have chest pain when you breathe deeply?",
            "Is the cough producing colored phlegm?",
            "How many days have you had the fever?"
        ],
        "conduta_sugerida": "Refer to health center or emergency depending on severity. If a child or elderly person with difficulty breathing: refer to emergency immediately.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Stroke",
        "descricao_clinica": "Suddenly one side of the face droops, arm or leg on one side loses strength, speech becomes slurred or the person cannot speak, mental confusion, very severe headache",
        "perguntas_chave": [
            "Did one side of the face or body suddenly become weak or droopy?",
            "Is the person having difficulty speaking or understanding?",
            "When did the symptoms start — how long ago exactly?",
            "Does the person have a history of high blood pressure or previous stroke?"
        ],
        "conduta_sugerida": "EMERGENCY. Call 192 (SAMU) immediately. Do not give anything by mouth. Every minute counts. Note the exact time symptoms began.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Heart attack",
        "descricao_clinica": "Strong chest pain that may spread to the left arm, jaw or back, cold sweats, shortness of breath, nausea, feeling of impending death",
        "perguntas_chave": [
            "Is the pain in the chest and spreading to the arm or jaw?",
            "Are you sweating cold and feeling short of breath?",
            "Do you have a history of heart problems or high blood pressure?",
            "How long have you had this pain?"
        ],
        "conduta_sugerida": "EMERGENCY. Call 192 (SAMU) immediately. Keep person seated or lying down. Do not leave them alone. Do not give medications without guidance.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Uncontrolled hypertension",
        "descricao_clinica": "Strong headache at the back of the neck, dizziness, blurred vision, ringing in the ears, person knows they have high blood pressure but is not taking medication or taking it incorrectly",
        "perguntas_chave": [
            "Have you been diagnosed with high blood pressure?",
            "Are you taking your blood pressure medication correctly?",
            "Do you have a headache at the back of your neck or blurred vision right now?",
            "Are you feeling dizzy or like you might faint?"
        ],
        "conduta_sugerida": "Refer urgently to health center to check blood pressure. If neurological symptoms present: emergency. Reinforce medication adherence.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Uncontrolled diabetes",
        "descricao_clinica": "Extreme thirst, frequent urination, blurred vision, extreme fatigue, wound that won't heal, sweet-smelling breath or person is confused and drowsy",
        "perguntas_chave": [
            "Have you been diagnosed with diabetes and are you managing it?",
            "Are you very thirsty and urinating frequently?",
            "Do you have a wound that is not healing?",
            "Are you confused, excessively sleepy or do you have unusual breath odor?"
        ],
        "conduta_sugerida": "Refer to health center. If person is confused, vomiting repeatedly or has strong sweet-smelling breath: emergency. Reinforce diet and medication adherence.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Pre-eclampsia",
        "descricao_clinica": "Pregnant woman with sudden swelling in hands and face, severe headache, vision with spots or blurred, pain in the upper abdomen above the navel, high blood pressure",
        "perguntas_chave": [
            "Are you pregnant? How many weeks along?",
            "Do you have sudden swelling in your face, hands or feet?",
            "Do you have a severe headache or blurred vision?",
            "Do you have pain in your upper abdomen above the navel?"
        ],
        "conduta_sugerida": "EMERGENCY for pregnant woman. Refer immediately to maternity or emergency room. Life-threatening for both mother and baby.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Severe childhood malnutrition",
        "descricao_clinica": "Very thin child with a swollen belly, thin brittle hair, skin with patches, extreme weakness, refusing to eat, weak or absent crying",
        "perguntas_chave": [
            "Is the child eating well?",
            "Has the child lost weight in recent months?",
            "Does the child have a swollen belly with very thin arms and legs?",
            "Is the child very weak, lacking energy or crying weakly?"
        ],
        "conduta_sugerida": "Refer urgently to health center or nutrition service. Register in SISVAN. Assess family social vulnerability.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Severe diarrhea with dehydration",
        "descricao_clinica": "Many liquid stools, vomiting, dry mouth, sunken eyes, child or elderly person not urinating for a long time, very weak or unresponsive",
        "perguntas_chave": [
            "How many times have they had diarrhea in the last few hours?",
            "Can they drink fluids or do they vomit everything?",
            "Are the eyes sunken and is the mouth dry?",
            "Are they urinating normally?"
        ],
        "conduta_sugerida": "Start oral rehydration solution or homemade rehydration drink immediately. If child or elderly with severe dehydration signs: refer to emergency urgently.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Asthma attack",
        "descricao_clinica": "Intense shortness of breath, wheezing in the chest, difficulty speaking full sentences, blue or purple lips or fingernails, person is frightened and leaning forward trying to breathe",
        "perguntas_chave": [
            "Have you been diagnosed with asthma?",
            "Can you speak full sentences without stopping to breathe?",
            "Are your lips or fingernails turning blue or purple?",
            "Have you used your inhaler today? How many times?"
        ],
        "conduta_sugerida": "If lips are blue or cannot speak: emergency immediately. Otherwise: refer to health center urgently. Guide on correct use of bronchodilator inhaler.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Snakebite",
        "descricao_clinica": "Person was bitten by a snake, intense pain at the site with swelling, purple or yellow patches, bleeding, dizziness, double vision or difficulty swallowing",
        "perguntas_chave": [
            "Did you see the snake or can you describe it?",
            "How long ago did the bite happen?",
            "Is there swelling, discoloration or bleeding at the site?",
            "Are you dizzy, seeing double or having difficulty swallowing?"
        ],
        "conduta_sugerida": "EMERGENCY. Call 192 (SAMU) or go to emergency with antivenom. Do not apply tourniquet or cut the wound. Immobilize the bitten limb.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Agrochemical poisoning",
        "descricao_clinica": "Rural worker with excessive saliva, tears, sweating, vomiting, diarrhea, tremors, seizure or difficulty breathing after contact with agricultural pesticide",
        "perguntas_chave": [
            "Did you have contact with pesticides today or in the past few days?",
            "Are you drooling excessively, sweating heavily or vomiting?",
            "Do you have tremors, muscle weakness or difficulty breathing?",
            "What product was used and how did contact occur?"
        ],
        "conduta_sugerida": "EMERGENCY. Remove person from contaminated area, remove clothing, wash with soap and water. Refer to emergency immediately with information about the pesticide used.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Cutaneous leishmaniasis",
        "descricao_clinica": "A painless skin wound that won't heal, with raised edges and reddish base, usually on exposed body parts. Person lives in rural or forested areas",
        "perguntas_chave": [
            "Do you have a wound on your skin that hasn't healed for weeks?",
            "Is the wound painless with raised edges?",
            "Do you live or work near a forest or rural area?",
            "Have you been bitten by a small pale sandfly?"
        ],
        "conduta_sugerida": "Refer to health center for investigation. Not an emergency, but requires specific treatment. Report to epidemiological surveillance.",
        "urgencia": 2,
        "regioes": ["BA", "MG", "MA", "PA", "PI", "PE", "CE", "RN", "PB", "AL", "SE", "TO"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Chagas disease",
        "descricao_clinica": "Person with swelling around one eye without apparent cause, fever, fatigue, facial swelling. Lives in a mud-brick house or has seen the kissing bug (large black insect with stripe)",
        "perguntas_chave": [
            "Do you live or have you lived in a mud-brick or adobe house?",
            "Have you seen or been bitten by a kissing bug (large black bug)?",
            "Is one eye swollen without any injury?",
            "Do you have fatigue, shortness of breath or frequent palpitations?"
        ],
        "conduta_sugerida": "Refer to health center for serology. In acute phase with symptoms: refer urgently. Investigate housing conditions and notify surveillance.",
        "urgencia": 2,
        "regioes": ["MG", "GO", "BA", "PI", "CE", "PE", "PB", "RN", "MA", "TO", "MT", "MS"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Schistosomiasis",
        "descricao_clinica": "Person who had contact with river or lake water, then felt skin itching, fever, abdominal pain, diarrhea. Chronic cases with a very swollen belly",
        "perguntas_chave": [
            "Did you have contact with river, lake or reservoir water recently?",
            "Did your skin itch after contact with the water?",
            "Do you have abdominal pain, diarrhea or blood in your stool?",
            "Is your belly swollen in an unusual way?"
        ],
        "conduta_sugerida": "Refer to health center for stool examination. Advise not to enter stagnant or river water in the area. Notify epidemiological surveillance.",
        "urgencia": 2,
        "regioes": ["MG", "BA", "PE", "AL", "SE", "PB", "RN", "CE", "MA", "ES", "RJ", "SP"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Hansen's disease (leprosy)",
        "descricao_clinica": "Lighter or reddish patches on the skin that don't itch and don't hurt, area with no feeling to heat or touch, weakness or tingling in hands and feet",
        "perguntas_chave": [
            "Do you have skin patches with no sensation (can't feel touch or heat)?",
            "Have the patches been there for more than 3 months?",
            "Do you have weakness, numbness or tingling in hands, feet or face?",
            "Does anyone close to you have or have had leprosy?"
        ],
        "conduta_sugerida": "Refer to health center for dermato-neurological assessment. Not an emergency, but requires long treatment. Investigate household contacts.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Bacterial meningitis",
        "descricao_clinica": "High fever, very severe headache, stiff neck (cannot touch chin to chest), purple or red spots on skin, sensitivity to light, may have projectile vomiting",
        "perguntas_chave": [
            "Do you have a very severe headache with a stiff neck?",
            "Have purple or red spots appeared on the skin?",
            "Is the light bothering your eyes?",
            "Are you confused, very drowsy or difficult to wake up?"
        ],
        "conduta_sugerida": "IMMEDIATE EMERGENCY. Call 192 (SAMU). Risk of death within hours. Do not wait — refer with maximum urgency to hospital.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Childhood febrile seizure",
        "descricao_clinica": "Young child with high fever who starts shaking all over, eyes rolling back, loses consciousness for a few minutes. Usually lasts less than 5 minutes",
        "perguntas_chave": [
            "How old is the child?",
            "Did the child have a fever before the seizure?",
            "Did the seizure last more than 5 minutes or did it happen again?",
            "Has the child had a seizure before?"
        ],
        "conduta_sugerida": "If seizure has stopped: refer urgently to health center. If it lasts more than 5 minutes or repeats: EMERGENCY. Lay on side, put nothing in the mouth. Control the fever.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Acute anxiety attack",
        "descricao_clinica": "Person with racing heart, shortness of breath, tingling hands, feeling of fainting or imminent death, trembling, sweating, very frightened for no apparent reason",
        "perguntas_chave": [
            "Have you had this kind of episode before?",
            "Do you have any known heart or lung condition?",
            "Have you been through a very stressful situation recently?",
            "Did the symptoms start suddenly without any apparent cause?"
        ],
        "conduta_sugerida": "Speak calmly and help them breathe slowly. If first episode or doubt about heart attack: refer to health center or emergency to rule out physical causes. Register for mental health follow-up.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },

    # ── Maternal health (5) ─────────────────────────────────────────────────
    {
        "nome": "High-risk pregnancy",
        "descricao_clinica": "Pregnant woman with diabetes, high blood pressure, under 15 or over 35 years old, twin pregnancy, history of previous miscarriages or a baby with a birth defect",
        "perguntas_chave": [
            "Are you attending prenatal care regularly?",
            "Do you have diabetes, high blood pressure or another chronic disease?",
            "Have you had a miscarriage or a baby with a problem before?",
            "Are you feeling the baby move normally?"
        ],
        "conduta_sugerida": "Refer to high-risk prenatal care at health center or reference maternity. Ensure at minimum monthly follow-up. Advise on warning signs.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Preterm labor",
        "descricao_clinica": "Pregnant woman with less than 37 weeks feeling regular contractions, back pain, pelvic pressure, leaking fluid or blood from the vagina",
        "perguntas_chave": [
            "How many weeks pregnant are you?",
            "Are you feeling contractions or cramps that come and go?",
            "Has any fluid, mucus or blood come out of your vagina?",
            "Do you have strong back pain or pressure pushing downward?"
        ],
        "conduta_sugerida": "EMERGENCY. Refer immediately to maternity. Do not let the woman walk. Call 192 (SAMU) if no safe transport is available.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Puerperal infection",
        "descricao_clinica": "Woman who gave birth a few days or weeks ago, with high fever, pain and bad smell at the birth or cesarean site, still very painful abdomen and foul-smelling discharge",
        "perguntas_chave": [
            "How many days ago did you give birth?",
            "Do you have a fever? How high?",
            "Is the birth site or scar red, producing pus or smelling bad?",
            "Do you have severe abdominal pain or chills?"
        ],
        "conduta_sugerida": "EMERGENCY. Puerperal infection can be fatal. Refer immediately to maternity or emergency room. Do not treat at home.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Mastitis",
        "descricao_clinica": "Breastfeeding woman with one breast red, hot, swollen and painful in a specific area, may have fever and general malaise. Sometimes forms a lump with pus",
        "perguntas_chave": [
            "Are you currently breastfeeding?",
            "Is one breast red, hot and very painful?",
            "Do you have fever or chills?",
            "Can you feel a hard lump or hardened area in the breast?"
        ],
        "conduta_sugerida": "Refer to health center. Advise not to stop breastfeeding from the affected side — it helps drain. If pus or high fever: urgent medical evaluation.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Postpartum depression",
        "descricao_clinica": "Mother who recently gave birth, very sad, crying for no reason, no desire to care for the baby, feeling unable to be a good mother, afraid of harming the baby or thinking of self-harm",
        "perguntas_chave": [
            "How long ago did you have the baby?",
            "Are you feeling very sad or hopeless?",
            "Are you able to care for and feel connected to your baby?",
            "Have you had thoughts of harming yourself or the baby?"
        ],
        "conduta_sugerida": "If thoughts of self-harm or harming the baby: refer urgently to mental health center or emergency. Otherwise: refer to health center with priority. Do not leave the mother alone. Involve family support.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Pediatric health (7) ────────────────────────────────────────────────
    {
        "nome": "Bronchiolitis",
        "descricao_clinica": "Baby or young child with cough, loud wheezing, fast or labored breathing, belly sucking in with each breath, too exhausted to breastfeed",
        "perguntas_chave": [
            "How old is the child?",
            "Is the child breathing very fast or with visible effort?",
            "Is the belly pulling in with each breath?",
            "Is the child able to breastfeed or eat normally?"
        ],
        "conduta_sugerida": "Refer to health center or emergency depending on severity. Babies under 3 months or with intense breathing effort: emergency immediately. Do not give cough syrup without guidance.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Whooping cough",
        "descricao_clinica": "Child with coughing fits that won't stop, coughing so hard they turn blue or vomit, at the end of the fit making a whooping sound while breathing in. Can last weeks",
        "perguntas_chave": [
            "Does the cough come in long fits that won't stop?",
            "Does the child turn blue or vomit during the coughing fit?",
            "Does the child make a whooping or gasping sound when breathing in after the fit?",
            "Is the child vaccinated with the pentavalent or DTP vaccine?"
        ],
        "conduta_sugerida": "Refer urgently to health center. In babies under 6 months: emergency, as they may stop breathing. Isolate from unvaccinated individuals. Notify surveillance.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Measles",
        "descricao_clinica": "Child with high fever, red eyes with discharge, cough, white spots inside the mouth, then a red rash starting on the face and spreading across the body",
        "perguntas_chave": [
            "Is the child vaccinated with the MMR or MMRV vaccine?",
            "Have red spots appeared starting from the face?",
            "Are the eyes red and producing discharge?",
            "Has the child had contact with someone who had a rash recently?"
        ],
        "conduta_sugerida": "MANDATORY IMMEDIATE NOTIFICATION. Refer to health center. Keep away from school and unvaccinated children. Investigate contacts and vaccinate those without proof of vaccination.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Chickenpox",
        "descricao_clinica": "Child with low fever and very itchy blisters spread across the body, starting on the trunk, with different stages at the same time: spots, fluid-filled blisters and crusts",
        "perguntas_chave": [
            "Are there blisters with fluid and crusts on the body at the same time?",
            "Are the blisters very itchy?",
            "Has the child had chickenpox before or received the varicella vaccine?",
            "Is there anyone immunocompromised at home (HIV, chemotherapy, steroids)?"
        ],
        "conduta_sugerida": "Advise school isolation until all lesions have crusted over. Do not give aspirin. If immunocompromised person or pregnant woman in the household: refer urgently to health center.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Impetigo",
        "descricao_clinica": "Child with sores or blisters on the skin that burst and form a honey-colored crust, mainly around the nose and mouth. Highly contagious",
        "perguntas_chave": [
            "Do the sores have a yellow or honey-colored crust on top?",
            "Do they appear mainly on the face or in areas of friction?",
            "Are other children at home or school with the same problem?",
            "Does the child scratch the sores and then touch other places?"
        ],
        "conduta_sugerida": "Refer to health center for topical or oral antibiotic. Advise frequent handwashing, keeping nails trimmed and not sharing towels.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Iron deficiency anemia in children",
        "descricao_clinica": "Pale child with pale gums, tongue and inner eyelids, easily tired, poor appetite, eating strange things like dirt or ice, growing below expected rate",
        "perguntas_chave": [
            "Are the child's gums or tongue pale?",
            "Does the child eat dirt, ice, chalk or other non-food items?",
            "Is the child growing and gaining weight normally?",
            "Does the child regularly eat meat, beans and dark leafy greens?"
        ],
        "conduta_sugerida": "Refer to health center for blood count. Advise iron-rich diet: meat, beans, dark leafy greens. Register in SISVAN. Verify correct use of iron supplement.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Mild dehydration",
        "descricao_clinica": "Child or adult with mild diarrhea or vomiting, slightly dry mouth, less urination than usual, no severe signs, still able to drink fluids",
        "perguntas_chave": [
            "Are they able to drink and keep fluids down?",
            "Are they still urinating, even if less than usual?",
            "Are their eyes and mouth dry?",
            "Are they feeling very weak or dizzy when standing up?"
        ],
        "conduta_sugerida": "Offer oral rehydration solution or homemade drink in small frequent sips. If no improvement in 4 hours or condition worsens: refer to health center. Monitor for signs of worsening.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Chronic diseases (6) ────────────────────────────────────────────────
    {
        "nome": "Heart failure",
        "descricao_clinica": "Person with shortness of breath that gets worse when lying down, very swollen legs and ankles, rapid weight gain, exhaustion with minimal effort, cough at night",
        "perguntas_chave": [
            "Are the legs swollen? Has the swelling gotten worse recently?",
            "Are you short of breath doing things that didn't used to cause that?",
            "Do you wake up at night short of breath or need more pillows to sleep?",
            "Have you gained weight quickly in the last few days for no apparent reason?"
        ],
        "conduta_sugerida": "Refer urgently to health center. If shortness of breath at rest or very severe swelling: emergency. Verify correct use of medications and salt restriction.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "COPD (chronic obstructive pulmonary disease)",
        "descricao_clinica": "Smoker or ex-smoker with chronic cough with phlegm, shortness of breath that gets progressively worse, chest wheezing, easy fatigue. May have acute flare-up with sudden worsening of breathlessness",
        "perguntas_chave": [
            "Do you smoke or have you smoked for many years?",
            "Is your shortness of breath getting progressively worse over time?",
            "Do you cough with phlegm every day?",
            "Has there been a sudden worsening of breathlessness or phlegm recently?"
        ],
        "conduta_sugerida": "Refer to health center for diagnosis and treatment. If acute flare-up with intense breathlessness: emergency. Strongly advise smoking cessation.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Chronic kidney disease",
        "descricao_clinica": "Person with diabetes or high blood pressure for many years, with swollen feet and legs, foamy or dark urine, very tired, poor appetite, unexplained body itching",
        "perguntas_chave": [
            "Have you had diabetes or high blood pressure for many years?",
            "Is your urine foamy, dark or has a strong smell?",
            "Do you have persistent swelling in your feet and legs?",
            "Do you have unexplained itching all over your body?"
        ],
        "conduta_sugerida": "Refer to health center for urine and kidney function tests. Do not use anti-inflammatory drugs. Reinforce tight blood pressure and blood sugar control to slow progression.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Epilepsy",
        "descricao_clinica": "Person with repeated seizures, which may involve full-body shaking, staring blankly for seconds, or repetitive movements without being conscious",
        "perguntas_chave": [
            "Have you had a seizure more than once in your life?",
            "Are you taking epilepsy medication regularly?",
            "Has there been a recent seizure? How long did it last?",
            "Was there any change in sleep, stress or a missed dose of medication?"
        ],
        "conduta_sugerida": "If in a seizure: lay on side, protect the head, do not restrain or put anything in the mouth. If it lasts more than 5 minutes: emergency. Outside a seizure: refer to health center for treatment adjustment.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Obesity with complications",
        "descricao_clinica": "Person with significant excess weight and difficulty walking, joint pain, shortness of breath with effort, associated high blood pressure and diabetes",
        "perguntas_chave": [
            "Do you have difficulty performing simple activities because of your weight?",
            "Do you have associated high blood pressure, diabetes or joint pain?",
            "Are you receiving follow-up care at the health center for your weight?",
            "Have you tried treatment or dieting before? With what result?"
        ],
        "conduta_sugerida": "Refer to health center for nutritional and physical activity follow-up. Investigate comorbidities. Assess referral to specialized service.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Hypothyroidism",
        "descricao_clinica": "Person very tired for no reason, gaining weight without eating more, always feeling cold even on warm days, dry skin, hair falling out, hoarse voice and slow thinking",
        "perguntas_chave": [
            "Are you always very tired even after resting well?",
            "Are you gaining weight without changing your diet?",
            "Do you feel cold when others don't, have dry skin or hair loss?",
            "Have you been diagnosed with a thyroid problem, or does anyone in your family have one?"
        ],
        "conduta_sugerida": "Refer to health center for TSH test. Not an emergency. If pregnant with suspicion: refer with priority as it affects baby's development.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Mental health (5) ───────────────────────────────────────────────────
    {
        "nome": "Suicide risk",
        "descricao_clinica": "Person saying they don't want to live anymore, that it would be better to be dead, saying goodbye to people, giving away prized possessions, isolated, with a previous attempt or plan",
        "perguntas_chave": [
            "Are you having thoughts of hurting yourself or not wanting to live anymore?",
            "Have you tried to hurt yourself before?",
            "Do you have a plan for how you would do it?",
            "Is there someone you trust nearby right now?"
        ],
        "conduta_sugerida": "NEVER minimize or question the person's feelings. Listen calmly and without judgment. Do not leave the person alone at any moment. Refer URGENTLY to mental health center or psychiatric emergency. Immediately involve family or support network.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Acute psychosis",
        "descricao_clinica": "Person hearing voices, seeing things that don't exist, behaving very strangely and aggressively, believing things that make no sense, not sleeping for days, may be aggressive",
        "perguntas_chave": [
            "Are you hearing voices or seeing things others cannot?",
            "Are you sleeping normally?",
            "Have you been diagnosed with schizophrenia or bipolar disorder?",
            "Are you taking your psychiatric medication regularly?"
        ],
        "conduta_sugerida": "Keep environment calm, do not challenge the person's beliefs. Refer urgently to mental health center or psychiatric emergency. Involve family. If there is a risk of aggression: call SAMU (192).",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Severe depression",
        "descricao_clinica": "Person very sad for weeks or months, unable to get out of bed, lost interest in everything they used to enjoy, poor appetite, sleeping too much or too little, feeling worthless",
        "perguntas_chave": [
            "How long have you been feeling this way?",
            "Are you able to carry out your daily activities as before?",
            "Have you lost interest in things you used to enjoy?",
            "Have you had thoughts that it would be better not to be here?"
        ],
        "conduta_sugerida": "Refer to health center or mental health center. If thoughts of suicide: maximum urgency. Do not leave person isolated. Involve family in care. Verify access to medications.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Alcohol abuse",
        "descricao_clinica": "Person drinking excessively and frequently, unable to stop, drinking in the morning, lost job or family due to drinking, has had seizures or shaking when trying to stop",
        "perguntas_chave": [
            "Do you drink alcohol every day or almost every day?",
            "Have you tried to stop and not been able to?",
            "Have you had tremors, heavy sweating or seizures when going without alcohol?",
            "Is drinking affecting your work, family or health?"
        ],
        "conduta_sugerida": "Refer to alcohol and drug mental health center (CAPS AD) or health center. CAUTION: sudden stopping after heavy use can cause seizures — do not advise abrupt cessation without medical supervision.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Generalized anxiety disorder",
        "descricao_clinica": "Person worrying excessively about everything all the time, unable to control it, with muscle tension, fatigue, difficulty concentrating, irritability and poor sleep for months",
        "perguntas_chave": [
            "Are you worrying a lot about many different things at the same time?",
            "Is this worry affecting your work, sleep or relationships?",
            "Do you have body tension, frequent headaches or tiredness?",
            "How long have you been feeling this way?"
        ],
        "conduta_sugerida": "Refer to health center or mental health center for assessment. Guide on breathing techniques and stress reduction. Register for ongoing mental health follow-up.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },

    # ── Rural skin conditions (4) ───────────────────────────────────────────
    {
        "nome": "Scabies",
        "descricao_clinica": "Intense itching that gets worse at night, small bumps or burrow lines on the skin between fingers, wrist, waist, armpit and genitals. Multiple people in the same household with itching",
        "perguntas_chave": [
            "Does the itching get much worse at night?",
            "Are there small bumps or lines between your fingers, on your wrists or waist?",
            "Are other people at home also itching?",
            "Have you tried any medication for the itching? Did it help?"
        ],
        "conduta_sugerida": "Refer to health center for treatment with permethrin or ivermectin. Treat EVERYONE in the household at the same time. Wash clothes and bedding in hot water. Not an emergency but very contagious.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Superficial fungal infection",
        "descricao_clinica": "Itchy skin patches with red borders and lighter center, or scaling between the toes with odor, or patches on the scalp with hair loss in spots",
        "perguntas_chave": [
            "Do the patches have a red border and lighter or scaly center?",
            "Is there itching and moisture between your toes?",
            "Do you work in a damp environment or wear closed shoes all day?",
            "Are the patches growing or spreading?"
        ],
        "conduta_sugerida": "Refer to health center for topical antifungal. Advise keeping skin dry, changing socks daily and not sharing personal hygiene items. Not an emergency.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Myiasis (wound fly larvae)",
        "descricao_clinica": "Wound with fly larvae inside, appearing as moving bumps, common in bedridden patients, domestic animals or people with open wounds that are poorly cared for in rural areas",
        "perguntas_chave": [
            "Does the wound have bumps that seem to move?",
            "Does the wound have a bad smell or discharge?",
            "Is the person bedridden or having difficulty with hygiene?",
            "What part of the body is the wound on?"
        ],
        "conduta_sugerida": "Refer to health center for larva removal by a professional. Do not try to remove at home by pressing — this can rupture the larvae and cause severe infection. Cover the wound with moist gauze.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Tungiasis (sand flea infestation)",
        "descricao_clinica": "Small white bump under the skin of the foot, usually at the nail edge or between the toes, causing itching and pain. Caused by a flea that burrows into the skin in dirt or soil areas",
        "perguntas_chave": [
            "Do you have a small white or black bump under the skin of your foot?",
            "Do you walk barefoot on dirt or in animal pens?",
            "Is the area red, itchy or producing pus?",
            "Do you have diabetes? (this increases the risk)"
        ],
        "conduta_sugerida": "Refer to health center for removal with a sterile needle by a professional. Do not pierce at home. Advise wearing footwear. If diabetic or signs of infection: higher priority.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },

    # ── Additional vector-borne diseases (2) ────────────────────────────────
    {
        "nome": "Yellow fever",
        "descricao_clinica": "High fever, headache, severe muscle aches, nausea. After a few days of improvement: return of fever with yellowing of skin and eyes, bleeding and dark vomiting",
        "perguntas_chave": [
            "Have you been in a forested area or yellow fever transmission zone recently?",
            "Is your yellow fever vaccination up to date?",
            "Are your skin or eyes turning yellow?",
            "Did you have a fever that improved then returned with dark vomiting or bleeding?"
        ],
        "conduta_sugerida": "EMERGENCY and IMMEDIATE NOTIFICATION. Refer to reference hospital. The toxic phase (yellow eyes + bleeding) has high mortality. Investigate vaccination of contacts.",
        "urgencia": 3,
        "regioes": ["AM", "PA", "MT", "RO", "AC", "RR", "AP", "TO", "MA", "GO", "MG", "BA", "ES", "SP", "PR", "RS", "SC"],
        "faixa_etaria": "all",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Hantavirus infection",
        "descricao_clinica": "Rural worker or person who entered an area with rodents, with fever, intense body aches and then rapidly worsening severe shortness of breath within hours",
        "perguntas_chave": [
            "Did you have contact with rodents, or their droppings or urine recently?",
            "Do you work or have you been in a rural area, grain silo, shed or closed barn?",
            "Did the shortness of breath start after a few days of fever and body aches?",
            "Is the shortness of breath getting much worse very rapidly?"
        ],
        "conduta_sugerida": "EMERGENCY. Hantavirus pulmonary syndrome has over 40% mortality. Refer immediately to reference hospital. MANDATORY NOTIFICATION. Investigate the exposure site.",
        "urgencia": 3,
        "regioes": ["SP", "SC", "RS", "PR", "MG", "GO", "MT", "MS", "RJ"],
        "faixa_etaria": "adult",
        "fonte": "MS-GVS",
        "validado_por": None,
    },

    # ── Emergency conditions (9) ────────────────────────────────────────────
    {
        "nome": "Suspected fracture",
        "descricao_clinica": "After a fall or trauma, limb with intense pain that gets worse with touch, visible deformity, rapid swelling, person cannot use the limb or bear weight",
        "perguntas_chave": [
            "Did you have a fall, impact or accident?",
            "Can you move or put weight on the affected limb?",
            "Does the limb look a different shape than normal?",
            "Did the pain get much worse with a light touch at the site?"
        ],
        "conduta_sugerida": "Immobilize the limb in the position it is in — do not try to straighten it. Refer to urgent care for X-ray. If bone is visible through the skin or limb is very cold with no pulse: emergency immediately.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Mild head injury",
        "descricao_clinica": "Person who hit their head, may have briefly lost consciousness, with headache, dizziness, nausea, mild confusion. May seem fine but can worsen later",
        "perguntas_chave": [
            "Did you lose consciousness, even for just a few seconds?",
            "Do you have a headache that is getting worse?",
            "Have you vomited after the injury?",
            "Are you confused, slurring your words or is one side weaker?"
        ],
        "conduta_sugerida": "Refer to health center or urgent care for assessment. If repeated vomiting, increasing confusion, one-sided weakness or different-sized pupils: EMERGENCY. Never leave alone in the first 24 hours.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Second-degree burn",
        "descricao_clinica": "Burn with blisters on the skin, red and very painful moist area. May be from hot water, fire, oil or chemicals. Large area or burns on face, hands or genitals are more serious",
        "perguntas_chave": [
            "What caused the burn?",
            "Are there blisters in the burned area?",
            "Is the burn on the face, neck, hands, feet or genitals?",
            "How large is the burned area approximately?"
        ],
        "conduta_sugerida": "Rinse with cool running water for 10–20 minutes. Do NOT use toothpaste, butter or any product. Cover with a clean cloth. Refer to health center. If face, hands, genitals or area larger than the palm of the hand: emergency.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Near-drowning",
        "descricao_clinica": "Person who nearly drowned and was rescued, may seem fine, but is coughing heavily, confused, breathing fast, or lost consciousness in the water",
        "perguntas_chave": [
            "Did the person lose consciousness or swallow a lot of water?",
            "Are they coughing intensely or short of breath after rescue?",
            "Are they confused or behaving differently than normal?",
            "How long were they struggling in or submerged in the water?"
        ],
        "conduta_sugerida": "EMERGENCY even if the person seems fine. Secondary drowning can occur hours later. Refer to hospital for at least 6 hours of observation. Do not release without medical evaluation.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Foreign body in child",
        "descricao_clinica": "Child who swallowed or put an object in the nose, ear or throat. May be choking, with sudden cough, difficulty breathing, drooling excessively or with a visible object",
        "perguntas_chave": [
            "Did you see or suspect the child swallowed or inhaled an object?",
            "Is the child having intense sudden coughing or difficulty breathing?",
            "Can the child swallow, speak or cry normally?",
            "What object might it have been: coin, button battery, small toy?"
        ],
        "conduta_sugerida": "If choking and cannot breathe: perform Heimlich maneuver immediately. BUTTON BATTERY swallowed: EMERGENCY — it burns the esophagus within hours. Any inhaled object: emergency. Do not try to remove blindly with a finger.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "child",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Anaphylactic shock",
        "descricao_clinica": "Shortly after an insect sting, food, medication or vaccine: severe hives, throat swelling, shortness of breath, blood pressure drop, extreme dizziness, fainting",
        "perguntas_chave": [
            "Did you have contact with something that may have caused an allergic reaction?",
            "How long after the contact did the symptoms start?",
            "Is there swelling in your throat, lips or tongue?",
            "Are you having difficulty breathing or nearly fainting?"
        ],
        "conduta_sugerida": "EMERGENCY. Call 192 (SAMU). Lay person flat with legs elevated. If the person has an epinephrine auto-injector: use it. Give nothing by mouth. Every minute counts.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Severe hypoglycemia",
        "descricao_clinica": "Diabetic on insulin or medication, shaking intensely, cold sweating, confused, with sudden intense hunger, may lose consciousness. Blood sugar is very low",
        "perguntas_chave": [
            "Do you have diabetes and use insulin or blood sugar medication?",
            "Are you shaking, sweating cold and suddenly very hungry?",
            "Did you eat normally before taking your medication or insulin?",
            "Can you swallow, or are you too confused?"
        ],
        "conduta_sugerida": "If conscious and able to swallow: give sugar, juice or regular soda immediately. Repeat in 15 minutes if no improvement. If unconscious: EMERGENCY — give nothing by mouth. Call 192.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Medication overdose",
        "descricao_clinica": "Person who took an excessive amount of medication, whether accidentally or intentionally, with extreme drowsiness, confusion, vomiting, slow or irregular breathing",
        "perguntas_chave": [
            "What medication was taken and how much?",
            "Was it accidental or intentional?",
            "How long ago did they take the medication?",
            "Are they conscious and responding normally?"
        ],
        "conduta_sugerida": "EMERGENCY. Call 192 (SAMU) and the Poison Control Center (0800 722 6001). Keep the medication packaging. If intentional: do not leave alone and urgently activate mental health support.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Cardiac arrest (recognition)",
        "descricao_clinica": "Person collapsed and unresponsive to calling, not breathing normally or breathing abnormally (like gurgling), no pulse. May have occurred after chest pain or without warning",
        "perguntas_chave": [
            "Is the person responding when you call their name and tap their shoulder?",
            "Are they breathing normally?",
            "Was there chest pain or fainting before they collapsed?",
            "How long have they been like this?"
        ],
        "conduta_sugerida": "IMMEDIATE EMERGENCY. Call 192 (SAMU) now. Start CPR: 30 hard compressions in the center of the chest + 2 rescue breaths — repeat until help arrives. Each minute without CPR reduces survival chances by 10%.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Common infections (8) ───────────────────────────────────────────────
    {
        "nome": "Cellulitis",
        "descricao_clinica": "Area of skin that is red, hot, swollen and painful and is spreading, usually after a wound, scratch or insect bite. May have fever",
        "perguntas_chave": [
            "Is the red area growing in the last few hours or days?",
            "Is it hot, firm and very painful to the touch?",
            "Was there a wound, bite or scratch before it appeared?",
            "Is there a fever or red lines spreading from the site?"
        ],
        "conduta_sugerida": "Refer to health center for antibiotic. If red lines spreading (lymphangitis), high fever or face affected: emergency. Mark the border of the redness to monitor expansion.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Skin abscess",
        "descricao_clinica": "Painful lump under the skin full of pus, hot and red, may have a white or yellow head at the center. May have fever and general malaise",
        "perguntas_chave": [
            "Is the lump getting bigger?",
            "Is it very painful and warm to touch?",
            "Do you have a fever along with it?",
            "Do you have diabetes or use steroids? (increases risk)"
        ],
        "conduta_sugerida": "Refer to health center for drainage by a professional. Do NOT squeeze at home — risk of spreading infection. If diabetic or abscess on the face: higher priority. Warm compresses help it ripen.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Infectious conjunctivitis",
        "descricao_clinica": "Red eye with yellow or green discharge that glues the eyelids together, tearing, feeling of sand in the eye. Very contagious. Can affect one or both eyes",
        "perguntas_chave": [
            "Are the eyes red with yellow or green discharge?",
            "Is the eyelid stuck shut in the morning when you wake up?",
            "Are other people nearby having the same problem?",
            "Is there intense pain, blurred vision or sensitivity to light?"
        ],
        "conduta_sugerida": "Refer to health center for antibiotic eye drops if bacterial. Advise frequent handwashing, not sharing towels. If intense pain or vision loss: ophthalmological emergency.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Acute otitis media",
        "descricao_clinica": "Child (or adult) with intense ear pain, fever, may have fluid coming out of the ear. Young child becomes very irritable, pulling the ear and crying intensely",
        "perguntas_chave": [
            "Do you have intense pain in the ear?",
            "Has fluid or pus come out of the ear?",
            "Do you have fever along with it?",
            "Have you had a cold or respiratory infection recently?"
        ],
        "conduta_sugerida": "Refer to health center for assessment and antibiotic if needed. Put nothing inside the ear. If pus draining, high fever or very young baby: priority at health center.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Streptococcal pharyngitis",
        "descricao_clinica": "Sudden intense sore throat, high fever, very red throat with white spots, difficulty swallowing, no cough or runny nose",
        "perguntas_chave": [
            "Did the sore throat start suddenly with a high fever?",
            "Are there white spots on the throat?",
            "Is there no cough or runny nose?",
            "Are there swollen and painful lymph nodes in your neck?"
        ],
        "conduta_sugerida": "Refer to health center for throat swab or rapid test. Treat with a full 10-day course of antibiotics to prevent rheumatic fever. Do not stop the antibiotic even if feeling better.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Urinary tract infection",
        "descricao_clinica": "Pain or burning when urinating, cloudy urine with strong smell, frequent urge to urinate with little output, pain in the lower abdomen. In elderly people may cause confusion",
        "perguntas_chave": [
            "Do you feel pain or burning when urinating?",
            "Is the urine cloudy, dark or has a different smell?",
            "Do you have back pain (at kidney level) or fever?",
            "In elderly: has the person become more confused than usual recently?"
        ],
        "conduta_sugerida": "Refer to health center for urine test and antibiotic. If fever, chills and back pain (pyelonephritis): refer urgently. Advise plenty of fluids.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Vaginal candidiasis",
        "descricao_clinica": "Woman with intense vaginal itching, white cottage-cheese-like discharge, burning when urinating, redness and swelling in the genital area",
        "perguntas_chave": [
            "Do you have intense itching in the vagina or vulva?",
            "Is the discharge white, lumpy and odorless?",
            "Have you recently used antibiotics or do you have diabetes?",
            "Have you had this problem before? Did you use any treatment?"
        ],
        "conduta_sugerida": "Refer to health center for confirmation and antifungal treatment. Advise proper hygiene without vaginal douching. If recurrent or pregnant: medical evaluation required.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Herpes zoster (shingles)",
        "descricao_clinica": "Intense pain followed by blisters that follow a strip on one side of the body or face, like a band. More common in elderly or immunocompromised people. Very painful",
        "perguntas_chave": [
            "Are the pain and blisters arranged in a strip on one side of the body?",
            "Did the pain start before the blisters appeared?",
            "Are you over 60 or do you have a weakened immune system?",
            "Are the blisters on the face near the eyes?"
        ],
        "conduta_sugerida": "Refer to health center urgently — antiviral medication is most effective in the first 72 hours. If blisters near the eyes: ophthalmological emergency. Not an emergency if mild and away from the eyes.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "elderly",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Special situations (4) ──────────────────────────────────────────────
    {
        "nome": "Suspected domestic violence",
        "descricao_clinica": "Woman, child or elderly person with injuries in unusual places, bite marks, cigarette burns, a story that doesn't match the injury, or showing fear when asked how they got hurt",
        "perguntas_chave": [
            "How did this injury happen? Does the story make sense?",
            "Are there injuries in different stages of healing?",
            "Does the person seem afraid to answer, or when the companion is nearby?",
            "Have there been similar situations before?"
        ],
        "conduta_sugerida": "Create an opportunity to speak privately with the person. Do not confront the aggressor. Document the injuries in detail. Contact social services, women's shelter or police as appropriate. File a violence notification report. Do not expose the victim to further risk.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Adult malnutrition",
        "descricao_clinica": "Adult or elderly person very thin without dieting, losing more than 10% of body weight in a few months, intense muscle weakness, wounds that won't heal, hair falling out",
        "perguntas_chave": [
            "How much weight have you lost and over how long?",
            "Are you able to eat normally or do you have difficulty?",
            "Do you have a chronic illness that might be causing the weight loss?",
            "Are you in a situation of food insecurity (not enough money for food)?"
        ],
        "conduta_sugerida": "Refer to health center to investigate the cause. Register in SISVAN. Contact social services if there is food insecurity. If bedridden elderly person with pressure sores: refer urgently.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adult",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Syncope (fainting)",
        "descricao_clinica": "Person who fainted for a few seconds to a few minutes with full rapid recovery. May have been preceded by dizziness, darkening of vision and paleness",
        "perguntas_chave": [
            "Did the person completely lose consciousness? For how long?",
            "Were there dizziness or darkening of vision before fainting?",
            "Were they standing for a long time, in heat, or had a fright?",
            "Do they have a history of heart problems?"
        ],
        "conduta_sugerida": "Lay person flat with legs elevated. If full recovery with no cardiac history: refer to health center. If it occurred during physical exertion, with seizure or for no apparent reason: emergency to investigate cardiac cause.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Suspected acute abdominal pain",
        "descricao_clinica": "Sudden severe abdominal pain, which may be located at a specific point, with or without fever, nausea and vomiting. May indicate appendicitis, kidney stone or other serious problems",
        "perguntas_chave": [
            "Did the pain start suddenly and is it getting worse?",
            "Is the pain fixed at one specific point in the abdomen?",
            "Do you have fever, vomiting, or is the abdomen rigid to the touch?",
            "Does the pain spread to the back, groin or shoulder?"
        ],
        "conduta_sugerida": "Refer to health center or urgent care for medical assessment. Do NOT give pain medication before evaluation (it can mask the diagnosis). If rigid abdomen, high fever or incapacitating pain: emergency.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "all",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
]

# ── Insert into database ────────────────────────────────────────────────────
vectors = [model.encode(c["descricao_clinica"]).tolist() for c in condicoes]

points = [
    PointStruct(id=i, vector=vectors[i], payload=condicoes[i])
    for i in range(len(condicoes))
]

client.points.upsert(COLECAO, points=points)

# ── Summary ─────────────────────────────────────────────────────────────────
contagem_fonte = Counter(c["fonte"] for c in condicoes)
validadas = sum(1 for c in condicoes if c["validado_por"] is not None)

print(f"\nTotal inserted: {len(condicoes)} conditions")
print(f"  - MS-CAB: {contagem_fonte.get('MS-CAB', 0)} conditions")
print(f"  - MS-GVS: {contagem_fonte.get('MS-GVS', 0)} conditions")
print(f"  - IA-pendente-validacao: {contagem_fonte.get('IA-pendente-validacao', 0)} conditions")
print(f"  - Validated by physician: {validadas} conditions")
