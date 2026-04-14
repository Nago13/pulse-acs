from actian_vectorai import VectorAIClient, DistanceMetric, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

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
    {
        "nome": "Dengue clássica",
        "descricao_clinica": "Febre alta que começa de repente, dor de cabeça forte, dor atrás dos olhos, dores no corpo e nas juntas, cansaço, pode aparecer manchas vermelhas na pele",
        "perguntas_chave": [
            "Há quantos dias está com febre?",
            "Tem dor atrás dos olhos ou na cabeça?",
            "Apareceu manchas vermelhas na pele?",
            "Tem vômito ou náusea?"
        ],
        "conduta_sugerida": "Orientar repouso, hidratação abundante e encaminhar para UBS para exame de sangue. Não usar AAS ou ibuprofeno. Monitorar sinais de alarme.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Dengue hemorrágica",
        "descricao_clinica": "Febre alta seguida de melhora e depois piora rápida, sangramento pelo nariz, gengiva ou na pele, barriga doendo, vômitos frequentes, pessoa fica muito fraca e com sono",
        "perguntas_chave": [
            "Teve febre que melhorou e depois piorou rápido?",
            "Está sangrando em algum lugar do corpo?",
            "Tem dor forte na barriga?",
            "Está muito fraco ou com dificuldade de ficar acordado?"
        ],
        "conduta_sugerida": "Encaminhar IMEDIATAMENTE para emergência. Risco de vida. Não dar medicamentos sem orientação médica.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Malária",
        "descricao_clinica": "Febre que vem e vai em ciclos, com calafrios fortes, tremedeira, suor excessivo, dor de cabeça e cansaço. Pessoa mora ou veio de área de mata ou rio",
        "perguntas_chave": [
            "A febre vem em horários certos ou em ciclos?",
            "Tem calafrios e tremedeira junto com a febre?",
            "Esteve em área de mata, rio ou outra cidade recentemente?",
            "Já teve malária antes?"
        ],
        "conduta_sugerida": "Encaminhar urgente para UBS ou unidade de saúde com teste rápido de malária. Não tratar sem confirmação laboratorial.",
        "urgencia": 2,
        "regioes": ["AM", "PA", "MT", "RO", "AC", "RR", "AP", "TO", "MA"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Leptospirose",
        "descricao_clinica": "Febre alta, dor de cabeça, dores no corpo especialmente nas panturrilhas, olhos avermelhados. Pessoa teve contato com água de enchente ou esgoto",
        "perguntas_chave": [
            "Teve contato com água de enchente, esgoto ou lama?",
            "Tem dor forte nas panturrilhas (batata da perna)?",
            "Os olhos estão vermelhos ou amarelados?",
            "Está com pouca urina ou urina escura?"
        ],
        "conduta_sugerida": "Encaminhar urgente para UBS ou emergência. Leptospirose pode evoluir rápido. Informar sobre o contato com água contaminada ao médico.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Zika",
        "descricao_clinica": "Febre baixa, manchas vermelhas pelo corpo com coceira, olhos avermelhados, dor nas juntas, inchaço nas mãos e pés. Gestante com esses sintomas é urgente",
        "perguntas_chave": [
            "Tem manchas vermelhas com coceira pelo corpo?",
            "Os olhos estão vermelhos e com coceira?",
            "É gestante ou pode estar grávida?",
            "Tem dor ou inchaço nas juntas?"
        ],
        "conduta_sugerida": "Se gestante: encaminhar com urgência para o pré-natal. Nos demais: encaminhar para UBS. Orientar sobre prevenção do Aedes aegypti.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Chikungunya",
        "descricao_clinica": "Febre alta de início súbito e dor muito forte nas juntas que pode impedir a pessoa de se mover, inchaço nas articulações, às vezes manchas na pele",
        "perguntas_chave": [
            "A dor nas juntas está tão forte que impede de se movimentar?",
            "A febre começou de repente?",
            "Tem inchaço nas mãos, pés ou joelhos?",
            "Outras pessoas na casa ou vizinhança com os mesmos sintomas?"
        ],
        "conduta_sugerida": "Encaminhar para UBS. Orientar repouso, hidratação e não usar AAS. A dor articular pode durar semanas ou meses.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Tuberculose",
        "descricao_clinica": "Tosse que não passa há mais de 3 semanas, às vezes com sangue, suor noturno, febre baixa no fim do dia, emagrecimento sem motivo e cansaço constante",
        "perguntas_chave": [
            "Há quantas semanas está com tosse?",
            "Já tossiu sangue ou catarro com sangue?",
            "Está emagrecendo sem fazer dieta?",
            "Tem alguém em casa com tuberculose ou tosse prolongada?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para coleta de escarro e raio-X. Orientar a cobrir a boca ao tossir. Investigar contatos domiciliares.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Pneumonia",
        "descricao_clinica": "Febre, tosse com catarro amarelo ou verde, dificuldade para respirar, dor no peito ao respirar, pessoa parece muito cansada e com falta de ar",
        "perguntas_chave": [
            "Está com dificuldade para respirar ou falta de ar?",
            "Tem dor no peito quando respira fundo?",
            "A tosse está produzindo catarro com cor?",
            "A febre está há quantos dias?"
        ],
        "conduta_sugerida": "Encaminhar para UBS ou emergência conforme gravidade. Se criança ou idoso com falta de ar: emergência imediata.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "AVC (derrame)",
        "descricao_clinica": "De repente a pessoa fica com um lado do rosto caído, braço ou perna sem força em um lado do corpo, fala enrolada ou não consegue falar, confusão mental, dor de cabeça muito forte",
        "perguntas_chave": [
            "Um lado do rosto ou do corpo ficou fraco ou caído de repente?",
            "Está com dificuldade para falar ou entender?",
            "Quando começaram os sintomas — faz quanto tempo?",
            "Tem histórico de pressão alta ou AVC anterior?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Ligar 192 (SAMU) imediatamente. Não dar nada pela boca. Cada minuto conta. Anotar a hora exata que os sintomas começaram.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto"
    },
    {
        "nome": "Infarto",
        "descricao_clinica": "Dor forte no peito que pode irradiar para o braço esquerdo, mandíbula ou costas, suor frio, falta de ar, náusea, sensação de morte iminente",
        "perguntas_chave": [
            "A dor está no peito e irradia para o braço ou mandíbula?",
            "Está suando frio e com falta de ar?",
            "Tem histórico de problemas no coração ou pressão alta?",
            "Há quanto tempo está com essa dor?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Ligar 192 (SAMU) imediatamente. Manter pessoa sentada ou deitada. Não deixar sozinha. Não dar remédios sem orientação.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto"
    },
    {
        "nome": "Hipertensão descompensada",
        "descricao_clinica": "Dor de cabeça forte na nuca, tontura, visão embaralhada, zumbido no ouvido, pessoa sabe que tem pressão alta mas está sem tomar remédio ou tomou errado",
        "perguntas_chave": [
            "Tem pressão alta diagnosticada?",
            "Está tomando os remédios da pressão corretamente?",
            "Tem dor de cabeça na nuca ou visão embaralhada agora?",
            "Está com tonturas ou sensação de desmaio?"
        ],
        "conduta_sugerida": "Encaminhar para UBS com urgência para aferir pressão. Se sintomas neurológicos presentes: emergência. Reforçar adesão ao tratamento.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto"
    },
    {
        "nome": "Diabetes descompensada",
        "descricao_clinica": "Muita sede, urina frequente, visão embaçada, cansaço extremo, ferida que não cicatriza, hálito com cheiro doce ou pessoa confusa e sonolenta",
        "perguntas_chave": [
            "Tem diabetes diagnosticada e está controlando?",
            "Está com muita sede e urinando muito?",
            "Tem alguma ferida que não está cicatrizando?",
            "Está confuso, com sono excessivo ou hálito estranho?"
        ],
        "conduta_sugerida": "Encaminhar para UBS. Se pessoa estiver confusa, com vômitos repetidos ou hálito adocicado forte: emergência. Reforçar adesão à dieta e medicação.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto"
    },
    {
        "nome": "Pré-eclâmpsia",
        "descricao_clinica": "Gestante com inchaço repentino nas mãos e rosto, dor de cabeça forte, visão com pontos ou borrada, dor na barriga acima do umbigo, pressão alta",
        "perguntas_chave": [
            "Está grávida? Com quantas semanas?",
            "Tem inchaço repentino no rosto, mãos ou pés?",
            "Está com dor de cabeça forte ou visão embaralhada?",
            "Tem dor na barriga acima do umbigo?"
        ],
        "conduta_sugerida": "EMERGÊNCIA para gestante. Encaminhar imediatamente para maternidade ou pronto-socorro. Risco de vida para mãe e bebê.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto"
    },
    {
        "nome": "Desnutrição infantil grave",
        "descricao_clinica": "Criança muito magra, com barriga inchada, cabelo fino e quebradiço, pele com manchas, fraqueza extrema, não quer comer, choro fraco ou ausente",
        "perguntas_chave": [
            "A criança está comendo bem?",
            "Perdeu peso nos últimos meses?",
            "Tem barriga inchada com braços e pernas muito finos?",
            "Está muito fraca, sem energia ou com choro fraco?"
        ],
        "conduta_sugerida": "Encaminhar urgente para UBS ou CAPS com nutrição infantil. Registrar no SISVAN. Avaliar situação de vulnerabilidade social da família.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "crianca"
    },
    {
        "nome": "Diarreia grave com desidratação",
        "descricao_clinica": "Muitas evacuações líquidas, vômito, boca seca, olhos fundos, criança ou idoso sem fazer xixi há muito tempo, muito fraco ou sem reagir",
        "perguntas_chave": [
            "Quantas vezes evacuou nas últimas horas?",
            "Está conseguindo beber água ou vomita tudo?",
            "Os olhos estão fundos e a boca seca?",
            "Está fazendo xixi normalmente?"
        ],
        "conduta_sugerida": "Iniciar soro caseiro ou soro de reidratação oral imediatamente. Se criança ou idoso com sinais graves de desidratação: emergência urgente.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Crise de asma",
        "descricao_clinica": "Falta de ar intensa, chiado no peito ao respirar, dificuldade para falar frases completas, lábios ou unhas arroxeadas, pessoa assustada e curvada para frente tentando respirar",
        "perguntas_chave": [
            "Tem asma diagnosticada?",
            "Está conseguindo falar frases completas sem parar?",
            "Os lábios ou unhas estão ficando roxos?",
            "Já usou a bombinha de inalação hoje? Quantas vezes?"
        ],
        "conduta_sugerida": "Se lábios roxos ou não consegue falar: emergência imediata. Caso contrário: encaminhar para UBS com urgência. Orientar uso correto do broncodilatador.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Picada de cobra",
        "descricao_clinica": "Pessoa foi picada por cobra, dor intensa no local com inchaço, manchas roxas ou amareladas, sangramento, tontura, visão dupla ou dificuldade para engolir",
        "perguntas_chave": [
            "Viu a cobra ou consegue descrever ela?",
            "Há quanto tempo foi a picada?",
            "Tem inchaço, manchas ou sangramento no local?",
            "Está com tontura, visão dupla ou dificuldade para engolir?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Ligar 192 (SAMU) ou levar para emergência com soro antiofídico. Não fazer torniquete, não cortar o local. Imobilizar o membro picado.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Intoxicação por agrotóxico",
        "descricao_clinica": "Trabalhador rural com saliva excessiva, lágrimas, suor, vômito, diarreia, tremores, convulsão ou dificuldade para respirar após contato com veneno agrícola",
        "perguntas_chave": [
            "Teve contato com agrotóxico hoje ou nos últimos dias?",
            "Está com salivação excessiva, suando muito ou com vômito?",
            "Tem tremores, fraqueza muscular ou dificuldade para respirar?",
            "Qual produto foi usado e como ocorreu o contato?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Remover da área contaminada, retirar roupas, lavar com água e sabão. Encaminhar para emergência imediatamente com informação do agrotóxico usado.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto"
    },
    {
        "nome": "Leishmaniose cutânea",
        "descricao_clinica": "Ferida na pele que não dói e não cicatriza, com borda elevada e fundo avermelhado, geralmente em partes expostas do corpo. Pessoa vive em área rural ou de mata",
        "perguntas_chave": [
            "Tem uma ferida que não cicatriza há semanas?",
            "A ferida é indolor com borda elevada?",
            "Mora ou trabalha próximo a mata ou área rural?",
            "Já foi picado por mosquito-palha (mosquito pequeno e claro)?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para investigação. Não é urgência, mas precisa de tratamento específico. Registrar o caso para vigilância epidemiológica.",
        "urgencia": 2,
        "regioes": ["BA", "MG", "MA", "PA", "PI", "PE", "CE", "RN", "PB", "AL", "SE", "TO"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Doença de Chagas",
        "descricao_clinica": "Pessoa com inchaço em um dos olhos sem causa aparente, febre, cansaço, inchaço no rosto. Vive em casa de pau-a-pique ou já viu o barbeiro (inseto preto com listra)",
        "perguntas_chave": [
            "Vive ou já morou em casa de pau-a-pique ou adobe?",
            "Já viu ou foi picado pelo barbeiro (percevejo grande, preto)?",
            "Tem inchaço em um dos olhos sem ter sofrido trauma?",
            "Está com cansaço, falta de ar ou palpitação frequente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para sorologia. Em fase aguda com sintomas: encaminhar com urgência. Investigar moradia e notificar à vigilância.",
        "urgencia": 2,
        "regioes": ["MG", "GO", "BA", "PI", "CE", "PE", "PB", "RN", "MA", "TO", "MT", "MS"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Esquistossomose",
        "descricao_clinica": "Pessoa que teve contato com água de rio ou lagoa, depois sentiu coceira na pele, febre, dor de barriga, diarreia. Casos crônicos com barriga inchada (barriga d'água)",
        "perguntas_chave": [
            "Teve contato com água de rio, lagoa ou açude recentemente?",
            "Sentiu coceira na pele após o contato com a água?",
            "Tem dor de barriga, diarreia ou sangue nas fezes?",
            "A barriga está inchada de forma diferente do normal?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para exame de fezes. Orientar sobre não entrar em água parada ou de rio na região. Notificar à vigilância epidemiológica.",
        "urgencia": 2,
        "regioes": ["MG", "BA", "PE", "AL", "SE", "PB", "RN", "CE", "MA", "ES", "RJ", "SP"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Hanseníase",
        "descricao_clinica": "Manchas na pele mais claras ou avermelhadas que não coçam e não doem, área sem sensibilidade ao calor ou toque, fraqueza ou formigamento nas mãos e pés",
        "perguntas_chave": [
            "Tem manchas na pele sem sensibilidade (não sente toque ou calor)?",
            "As manchas existem há mais de 3 meses?",
            "Tem fraqueza, dormência ou formigamento nas mãos, pés ou rosto?",
            "Alguém próximo tem ou teve hanseníase?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para avaliação dermatoneurológica. Não é urgência, mas requer tratamento longo. Investigar contatos domiciliares.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Meningite bacteriana",
        "descricao_clinica": "Febre alta, dor de cabeça muito forte, rigidez no pescoço (não consegue encostar o queixo no peito), manchas roxas na pele, sensibilidade à luz, pode ter vômitos em jato",
        "perguntas_chave": [
            "Tem dor de cabeça muito forte com rigidez no pescoço?",
            "Apareceu manchas roxas ou vermelhas na pele?",
            "A luz está incomodando muito os olhos?",
            "Está confuso, muito sonolento ou difícil de acordar?"
        ],
        "conduta_sugerida": "EMERGÊNCIA IMEDIATA. Ligar 192 (SAMU). Risco de morte em horas. Não esperar — encaminhar com urgência máxima para hospital.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
    {
        "nome": "Convulsão febril infantil",
        "descricao_clinica": "Criança pequena com febre alta que começa a tremer o corpo todo, olhos virados, fica inconsciente por alguns minutos. Geralmente dura menos de 5 minutos",
        "perguntas_chave": [
            "Qual a idade da criança?",
            "A criança estava com febre antes da convulsão?",
            "A convulsão durou mais de 5 minutos ou voltou a convulsionar?",
            "Já teve convulsão antes?"
        ],
        "conduta_sugerida": "Se convulsão parou: encaminhar para UBS urgente. Se durar mais de 5 minutos ou repetir: EMERGÊNCIA. Deitar de lado, não colocar nada na boca. Controlar a febre.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "crianca"
    },
    {
        "nome": "Crise de ansiedade aguda",
        "descricao_clinica": "Pessoa com coração acelerado, falta de ar, formigamento nas mãos, sensação de desmaio ou de morte iminente, tremor, sudorese, muito assustada mesmo sem causa aparente",
        "perguntas_chave": [
            "Já teve esse tipo de crise antes?",
            "Tem algum problema de saúde no coração ou pulmão?",
            "Passou por alguma situação muito estressante recentemente?",
            "Os sintomas começaram de repente sem causa aparente?"
        ],
        "conduta_sugerida": "Falar com calma, ajudar a respirar devagar. Se primeira crise ou dúvida com infarto: encaminhar para UBS ou emergência para descartar causas físicas. Registrar para acompanhamento de saúde mental.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos"
    },
]

vectors = [model.encode(c["descricao_clinica"]).tolist() for c in condicoes]

points = [
    PointStruct(id=i, vector=vectors[i], payload=condicoes[i])
    for i in range(len(condicoes))
]

client.points.upsert(COLECAO, points=points)
print(f"{len(condicoes)} condições inseridas com sucesso.")
