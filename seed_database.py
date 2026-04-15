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
    # ── 25 condições originais ──────────────────────────────────────────────
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "adulto",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
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
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
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
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },

    # ── Saúde materna (5) ───────────────────────────────────────────────────
    {
        "nome": "Pré-natal de risco",
        "descricao_clinica": "Gestante com diabetes, pressão alta, idade abaixo de 15 ou acima de 35 anos, gravidez gemelar, histórico de perdas anteriores ou bebê anterior com malformação",
        "perguntas_chave": [
            "Está fazendo pré-natal regularmente?",
            "Tem diabetes, pressão alta ou outra doença crônica?",
            "Já perdeu gravidez ou teve bebê com problema antes?",
            "Está sentindo o bebê se mover normalmente?"
        ],
        "conduta_sugerida": "Encaminhar para pré-natal de alto risco na UBS ou maternidade de referência. Garantir acompanhamento mensal no mínimo. Orientar sinais de alarme.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Trabalho de parto prematuro",
        "descricao_clinica": "Gestante com menos de 37 semanas sentindo contrações regulares, dor nas costas, pressão na pelve, saída de líquido ou sangue pela vagina",
        "perguntas_chave": [
            "Com quantas semanas de gravidez está?",
            "Está sentindo contrações ou cólicas que vêm e vão?",
            "Saiu líquido, muco ou sangue pela vagina?",
            "Tem dor forte nas costas ou pressão para baixo?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Encaminhar imediatamente para maternidade. Não deixar a gestante caminhar. Ligar 192 (SAMU) se não houver transporte seguro.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Infecção puerperal",
        "descricao_clinica": "Mulher que teve bebê há poucos dias ou semanas, com febre alta, dor e mau cheiro no local do parto ou da cesárea, barriga ainda muito dolorida e lóquios com cheiro ruim",
        "perguntas_chave": [
            "Há quantos dias foi o parto?",
            "Está com febre? Quanto?",
            "O local do parto ou da cicatriz está com vermelhidão, pus ou cheiro ruim?",
            "Está com dor forte na barriga ou calafrios?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Infecção puerperal pode ser fatal. Encaminhar imediatamente para maternidade ou emergência. Não tratar em casa.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Mastite",
        "descricao_clinica": "Mulher amamentando com mama vermelha, quente, inchada e dolorida em uma área específica, pode ter febre e mal-estar. Às vezes forma nódulo com pus",
        "perguntas_chave": [
            "Está amamentando atualmente?",
            "A mama está vermelha, quente e muito dolorida?",
            "Tem febre ou calafrios?",
            "Consegue sentir um caroço ou área endurecida na mama?"
        ],
        "conduta_sugerida": "Encaminhar para UBS. Orientar a não parar de amamentar do lado afetado — ajuda a drenar. Se houver pus ou febre alta: urgência para avaliação médica.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Depressão pós-parto",
        "descricao_clinica": "Mãe que teve bebê recentemente, muito triste, chorando sem motivo, sem vontade de cuidar do bebê, sentindo que não consegue ser boa mãe, com medo de machucar o bebê ou pensando em se machucar",
        "perguntas_chave": [
            "Há quanto tempo teve o bebê?",
            "Está se sentindo muito triste ou sem esperança?",
            "Está conseguindo cuidar e se sentir ligada ao bebê?",
            "Teve pensamento de se machucar ou machucar o bebê?"
        ],
        "conduta_sugerida": "Se pensamento de se machucar ou machucar o bebê: encaminhar urgente para CAPS ou emergência. Nos demais: encaminhar para UBS com prioridade. Não deixar a mãe sozinha. Acionar rede familiar.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Saúde infantil (7) ──────────────────────────────────────────────────
    {
        "nome": "Bronquiolite",
        "descricao_clinica": "Bebê ou criança pequena com tosse, chiado forte no peito, respiração rápida e com esforço, barriga puxando para dentro ao respirar, muito cansada para mamar",
        "perguntas_chave": [
            "Qual a idade da criança?",
            "Está respirando muito rápido ou com esforço visível?",
            "A barriga está puxando para dentro ao respirar?",
            "Está conseguindo mamar ou se alimentar normalmente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS ou emergência conforme gravidade. Bebê com menos de 3 meses ou com sinais de esforço respiratório intenso: emergência imediata. Não dar xarope sem orientação.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Coqueluche",
        "descricao_clinica": "Criança com tosse em crises que não param, tosse tão forte que fica roxa ou vomita, ao fim da crise faz um barulho como 'guincho' ao respirar. Pode durar semanas",
        "perguntas_chave": [
            "A tosse vem em crises longas e sem parar?",
            "A criança fica roxa ou vomita durante a tosse?",
            "Faz um barulho de guincho ao respirar fundo depois da crise?",
            "Está vacinada com pentavalente ou DTP?"
        ],
        "conduta_sugerida": "Encaminhar para UBS com urgência. Em bebês menores de 6 meses: emergência, pois pode parar de respirar. Isolar de outros não vacinados. Notificar vigilância.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Sarampo",
        "descricao_clinica": "Criança com febre alta, olhos vermelhos com secreção, tosse, manchas brancas dentro da boca e depois manchas vermelhas que começam no rosto e se espalhram pelo corpo",
        "perguntas_chave": [
            "Está vacinada com a vacina tríplice viral ou tetra viral?",
            "Apareceram manchas vermelhas começando no rosto?",
            "Os olhos estão vermelhos e com secreção?",
            "Teve contato com alguém com erupção cutânea recentemente?"
        ],
        "conduta_sugerida": "NOTIFICAÇÃO COMPULSÓRIA IMEDIATA. Encaminhar para UBS. Isolar da escola e de crianças não vacinadas. Investigar contatos e vacinar comunicantes sem comprovante.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Varicela (catapora)",
        "descricao_clinica": "Criança com febre baixa e bolhas que coçam muito espalhadas pelo corpo, começando no tronco, com diferentes estágios: manchas, bolinhas cheias de líquido e crostas",
        "perguntas_chave": [
            "Tem bolhas com líquido e crostas ao mesmo tempo no corpo?",
            "As bolhas coçam muito?",
            "Já teve catapora antes ou tomou a vacina?",
            "Tem alguém imunocomprometido em casa (HIV, quimio, corticoide)?"
        ],
        "conduta_sugerida": "Orientar isolamento escolar até todas as lesões virarem crosta. Não dar AAS. Se imunocomprometido ou grávida na casa: encaminhar com urgência para UBS.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Impetigo",
        "descricao_clinica": "Criança com feridas ou bolhas na pele que arrebentam e formam crosta amarela como mel, principalmente no rosto, ao redor do nariz e boca. Muito contagioso",
        "perguntas_chave": [
            "As feridas têm crosta amarela ou dourada por cima?",
            "Aparecem principalmente no rosto ou em áreas de atrito?",
            "Outras crianças na casa ou escola com o mesmo problema?",
            "A criança coça as feridas e depois toca em outros lugares?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para tratamento com antibiótico tópico ou oral. Orientar higiene das mãos, cortar unhas curtas e não compartilhar toalhas.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Anemia ferropriva infantil",
        "descricao_clinica": "Criança pálida nas gengivas, na língua e nas pálpebras, cansada fácil, sem apetite, come coisas estranhas como terra ou gelo, crescimento abaixo do esperado",
        "perguntas_chave": [
            "A criança está pálida nas gengivas ou na língua?",
            "Come terra, gelo, tijolo ou outras coisas não comestíveis?",
            "Está crescendo e ganhando peso normalmente?",
            "Come carne, feijão e folhas verdes regularmente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para hemograma. Orientar alimentação rica em ferro: carne, feijão, folhas escuras. Registrar no SISVAN. Verificar uso correto do ferro suplementar.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Desidratação leve",
        "descricao_clinica": "Criança ou adulto com diarreia ou vômito leve, boca um pouco seca, menos xixi que o normal, sem sinais graves, ainda consegue beber líquidos",
        "perguntas_chave": [
            "Está conseguindo beber e segurar líquidos?",
            "Está fazendo xixi, mesmo que menos que o normal?",
            "Os olhos e a boca estão secos?",
            "Está se sentindo muito fraco ou com tontura ao levantar?"
        ],
        "conduta_sugerida": "Oferecer soro de reidratação oral ou soro caseiro em pequenos goles frequentes. Se piorar ou não melhorar em 4 horas: encaminhar para UBS. Monitorar sinais de piora.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Doenças crônicas (6) ────────────────────────────────────────────────
    {
        "nome": "Insuficiência cardíaca",
        "descricao_clinica": "Pessoa com falta de ar que piora deitada, pernas e tornozelos muito inchados, ganho rápido de peso, cansaço ao menor esforço, tosse à noite",
        "perguntas_chave": [
            "As pernas estão inchadas? O inchaço ficou pior ultimamente?",
            "Está com falta de ar para coisas que antes não causavam?",
            "Acorda à noite com falta de ar ou precisa de mais travesseiros?",
            "Ganhou peso rápido nos últimos dias sem razão aparente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS com urgência. Se falta de ar em repouso ou inchaço muito grave: emergência. Verificar uso correto dos medicamentos e restrição de sal.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "DPOC (doença pulmonar obstrutiva crônica)",
        "descricao_clinica": "Fumante ou ex-fumante com tosse crônica com catarro, falta de ar que piora com o tempo, chiado no peito, cansaço fácil. Pode ter crise aguda com piora súbita da falta de ar",
        "perguntas_chave": [
            "Fuma ou fumou por muitos anos?",
            "A falta de ar está piorando progressivamente?",
            "Tem tosse com catarro todos os dias?",
            "Houve piora súbita da falta de ar ou do catarro recentemente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para diagnóstico e tratamento. Se crise aguda com falta de ar intensa: emergência. Orientar fortemente sobre cessação do tabagismo.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Doença renal crônica",
        "descricao_clinica": "Pessoa com diabetes ou pressão alta há anos, com inchaço nos pés e pernas, urina espumosa ou escura, muito cansada, sem apetite, com coceira no corpo sem causa aparente",
        "perguntas_chave": [
            "Tem diabetes ou pressão alta há muitos anos?",
            "A urina está espumosa, escura ou com cheiro forte?",
            "Tem inchaço persistente nos pés e pernas?",
            "Está com coceira no corpo sem causa aparente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para exame de urina e função renal. Não usar anti-inflamatórios. Reforçar controle rigoroso da pressão e glicemia para retardar progressão.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Epilepsia",
        "descricao_clinica": "Pessoa com histórico de convulsões repetidas, que podem ser com tremor do corpo todo, olhar fixo e parado por segundos, ou movimentos repetitivos sem estar consciente",
        "perguntas_chave": [
            "Já teve convulsão mais de uma vez na vida?",
            "Está tomando medicamento para epilepsia regularmente?",
            "Houve convulsão recente? Quanto tempo durou?",
            "Houve alguma mudança no sono, estresse ou esquecimento do remédio?"
        ],
        "conduta_sugerida": "Se em crise: deitar de lado, proteger a cabeça, não segurar nem colocar nada na boca. Se durar mais de 5 minutos: emergência. Fora da crise: encaminhar para UBS para ajuste de tratamento.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Obesidade com complicações",
        "descricao_clinica": "Pessoa com muito excesso de peso e dificuldade para caminhar, dor nas articulações, falta de ar ao esforço, pressão alta e diabetes associadas",
        "perguntas_chave": [
            "Tem dificuldade para realizar atividades simples por causa do peso?",
            "Tem pressão alta, diabetes ou dor nas articulações associadas?",
            "Está em acompanhamento na UBS para o peso?",
            "Já tentou tratamento ou dieta antes? Com qual resultado?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para programa de acompanhamento nutricional e atividade física. Investigar comorbidades. Avaliar indicação de encaminhamento para serviço especializado.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Hipotireoidismo",
        "descricao_clinica": "Pessoa muito cansada sem motivo, engordando sem comer mais, com frio mesmo em dia quente, pele seca, cabelo caindo, voz mais grossa e raciocínio lento",
        "perguntas_chave": [
            "Está sempre com muito cansaço mesmo descansando bem?",
            "Está engordando sem mudar a alimentação?",
            "Tem frio quando os outros não têm, pele seca ou queda de cabelo?",
            "Já teve diagnóstico de problema na tireoide ou alguém na família tem?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para exame de TSH. Não é urgência. Se gestante com suspeita: encaminhar com prioridade pois afeta o desenvolvimento do bebê.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Saúde mental (5) ────────────────────────────────────────────────────
    {
        "nome": "Risco de suicídio",
        "descricao_clinica": "Pessoa falando que não quer mais viver, que seria melhor estar morta, se despedindo das pessoas, doando objetos queridos, isolada, com plano ou tentativa anterior",
        "perguntas_chave": [
            "Está tendo pensamentos de se machucar ou de não querer mais viver?",
            "Já tentou se machucar antes?",
            "Tem algum plano de como faria isso?",
            "Tem alguém de confiança por perto agora?"
        ],
        "conduta_sugerida": "NUNCA minimizar ou questionar. Fazer escuta ativa com calma e sem julgamento. Não deixar a pessoa sozinha em nenhum momento. Encaminhar URGENTE para CAPS ou emergência psiquiátrica. Acionar família ou rede de apoio imediatamente.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Psicose aguda",
        "descricao_clinica": "Pessoa ouvindo vozes, vendo coisas que não existem, com comportamento muito estranho e agitado, acreditando em coisas sem sentido, sem dormir há dias, podendo ser agressiva",
        "perguntas_chave": [
            "Está ouvindo vozes ou vendo coisas que os outros não veem?",
            "Está dormindo normalmente?",
            "Tem diagnóstico de esquizofrenia ou transtorno bipolar?",
            "Está tomando algum medicamento psiquiátrico regularmente?"
        ],
        "conduta_sugerida": "Manter ambiente calmo, não confrontar as crenças da pessoa. Encaminhar para CAPS ou emergência psiquiátrica com urgência. Acionar família. Se agitação com risco de agressão: acionar SAMU.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Depressão grave",
        "descricao_clinica": "Pessoa muito triste há semanas ou meses, sem conseguir sair da cama, perdeu o interesse em tudo que gostava, sem apetite, dormindo demais ou de menos, se sentindo inútil",
        "perguntas_chave": [
            "Há quanto tempo está se sentindo assim?",
            "Consegue realizar as atividades do dia a dia como antes?",
            "Perdeu o interesse em coisas que antes davam prazer?",
            "Tem pensamentos de que seria melhor não estar aqui?"
        ],
        "conduta_sugerida": "Encaminhar para UBS ou CAPS. Se houver pensamentos de suicídio: urgência máxima. Não deixar a pessoa isolada. Envolver família no cuidado. Verificar acesso a medicamentos.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Abuso de álcool",
        "descricao_clinica": "Pessoa bebendo em excesso com frequência, não consegue parar, bebe pela manhã, perdeu emprego ou família por causa da bebida, já teve convulsão ou tremores ao tentar parar",
        "perguntas_chave": [
            "Bebe todos os dias ou quase todos os dias?",
            "Já tentou parar e não conseguiu?",
            "Já teve tremores, suor intenso ou convulsão quando ficou sem beber?",
            "A bebida está afetando o trabalho, família ou saúde?"
        ],
        "conduta_sugerida": "Encaminhar para CAPS AD (álcool e drogas) ou UBS para acompanhamento. ATENÇÃO: parar abruptamente após uso pesado pode causar convulsão — não orientar parada abrupta sem acompanhamento médico.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Transtorno de ansiedade generalizada",
        "descricao_clinica": "Pessoa preocupada excessivamente com tudo o tempo todo, sem conseguir controlar, com tensão muscular, cansaço, dificuldade de concentração, irritabilidade e sono ruim há meses",
        "perguntas_chave": [
            "Está se preocupando muito com várias coisas ao mesmo tempo?",
            "Essa preocupação está afetando o trabalho, o sono ou os relacionamentos?",
            "Sente tensão no corpo, dor de cabeça ou cansaço frequente?",
            "Há quanto tempo está se sentindo assim?"
        ],
        "conduta_sugerida": "Encaminhar para UBS ou CAPS para avaliação. Orientar sobre técnicas de respiração e redução de estresse. Registrar para acompanhamento longitudinal de saúde mental.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },

    # ── Doenças de pele rurais (4) ──────────────────────────────────────────
    {
        "nome": "Escabiose (sarna)",
        "descricao_clinica": "Coceira intensa que piora à noite, pequenas bolinhas ou riscos na pele entre os dedos, pulso, cintura, axila e genitais. Várias pessoas da mesma casa com coceira",
        "perguntas_chave": [
            "A coceira piora muito à noite?",
            "Tem risquinhos ou bolinhas entre os dedos, no pulso ou na cintura?",
            "Outras pessoas na casa estão com coceira também?",
            "Já usou algum remédio para a coceira? Melhorou?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para tratamento com permetrina ou ivermectina. Tratar TODOS da casa ao mesmo tempo. Lavar roupas e roupas de cama com água quente. Não é urgência mas é muito contagiosa.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Micose superficial",
        "descricao_clinica": "Manchas na pele com coceira, bordas avermelhadas e centro mais claro, ou entre os dedos dos pés com descamação e cheiro, ou no couro cabeludo com queda de cabelo em áreas",
        "perguntas_chave": [
            "As manchas têm borda vermelha e centro mais claro ou descamam?",
            "Tem coceira e umidade entre os dedos dos pés?",
            "Trabalha em ambiente úmido ou usa calçado fechado o dia todo?",
            "As manchas estão crescendo ou se espalhando?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para antifúngico tópico. Orientar manter pele seca, trocar meias diariamente e não compartilhar objetos de higiene. Não é urgência.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Miíase",
        "descricao_clinica": "Ferida com larvas de mosca dentro, aparecendo como bolinhas que se movem, comum em pessoas acamadas, animais domésticos ou com feridas abertas mal cuidadas em área rural",
        "perguntas_chave": [
            "Tem ferida na pele com bolinhas que parecem se mover?",
            "A ferida tem cheiro ruim ou secreção?",
            "A pessoa fica acamada ou tem dificuldade de cuidar da higiene?",
            "Em que parte do corpo está a ferida?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para remoção das larvas por profissional. Não tentar remover em casa com pressão pois pode romper as larvas e causar infecção grave. Cobrir a ferida com gaze úmida.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Tungíase (bicho-de-pé)",
        "descricao_clinica": "Pequena bolinha branca debaixo da pele do pé, geralmente na borda da unha ou entre os dedos, que coça e dói. Causada por pulga que entra na pele em solo de terra",
        "perguntas_chave": [
            "Tem uma bolinha branca ou preta debaixo da pele do pé?",
            "Anda descalço em terra batida ou curral?",
            "A área está vermelha, coçando ou com pus?",
            "Tem diabetes? (pois complica mais)"
        ],
        "conduta_sugerida": "Encaminhar para UBS para remoção com agulha estéril por profissional. Não furar em casa. Orientar uso de calçados. Se diabético ou com sinais de infecção: maior prioridade.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },

    # ── Vetores adicionais (2) ──────────────────────────────────────────────
    {
        "nome": "Febre amarela",
        "descricao_clinica": "Febre alta, dor de cabeça, dores musculares fortes, náusea. Após alguns dias de melhora: retorno da febre com pele e olhos amarelos, sangramento e vômito escuro",
        "perguntas_chave": [
            "Esteve em área de mata ou de transmissão de febre amarela recentemente?",
            "Está com a vacina de febre amarela em dia?",
            "A pele ou os olhos estão ficando amarelos?",
            "Teve melhora da febre e depois piora com vômito escuro ou sangramento?"
        ],
        "conduta_sugerida": "EMERGÊNCIA e NOTIFICAÇÃO IMEDIATA. Encaminhar para hospital de referência. A fase tóxica (olhos amarelos + sangramento) tem alta mortalidade. Investigar vacinação dos contatos.",
        "urgencia": 3,
        "regioes": ["AM", "PA", "MT", "RO", "AC", "RR", "AP", "TO", "MA", "GO", "MG", "BA", "ES", "SP", "PR", "RS", "SC"],
        "faixa_etaria": "todos",
        "fonte": "MS-GVS",
        "validado_por": None,
    },
    {
        "nome": "Hantavirose",
        "descricao_clinica": "Trabalhador rural ou pessoa que entrou em local com roedores, com febre, dor muscular intensa e depois falta de ar grave que piora rapidamente em poucas horas",
        "perguntas_chave": [
            "Teve contato com roedores, fezes ou urina de camundongos recentemente?",
            "Trabalha ou esteve em área rural, silo, paiol ou galpão fechado?",
            "A falta de ar começou após alguns dias de febre e dor muscular?",
            "A falta de ar está piorando muito rapidamente?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Hantavirose pulmonar tem mortalidade acima de 40%. Encaminhar imediatamente para hospital de referência. NOTIFICAÇÃO COMPULSÓRIA. Investigar local de exposição.",
        "urgencia": 3,
        "regioes": ["SP", "SC", "RS", "PR", "MG", "GO", "MT", "MS", "RJ"],
        "faixa_etaria": "adulto",
        "fonte": "MS-GVS",
        "validado_por": None,
    },

    # ── Urgências (9) ───────────────────────────────────────────────────────
    {
        "nome": "Fratura suspeita",
        "descricao_clinica": "Após queda ou trauma, membro com dor intensa que piora ao toque, deformidade visível, inchaço rápido, pessoa não consegue usar o membro ou apoiar peso",
        "perguntas_chave": [
            "Sofreu queda, pancada ou acidente?",
            "Consegue mover ou apoiar peso no membro afetado?",
            "O membro está com formato diferente do normal?",
            "A dor piorou muito ao toque leve no local?"
        ],
        "conduta_sugerida": "Imobilizar o membro na posição que está — não tentar endireitar. Encaminhar para urgência com radiografia. Se osso aparecer pela pele ou membro muito frio e sem pulso: emergência imediata.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Traumatismo craniano leve",
        "descricao_clinica": "Pessoa que bateu a cabeça, pode ter perdido a consciência por breve momento, com dor de cabeça, tontura, náusea, confusão leve. Pode parecer bem mas piorar depois",
        "perguntas_chave": [
            "Perdeu a consciência, mesmo que por alguns segundos?",
            "Está com dor de cabeça que está piorando?",
            "Vomitou após o trauma?",
            "Está confuso, com fala arrastada ou um lado mais fraco?"
        ],
        "conduta_sugerida": "Encaminhar para UBS ou urgência para avaliação. Se vômito repetido, confusão crescente, um lado fraco ou pupila diferente: EMERGÊNCIA. Nunca deixar sozinho nas primeiras 24 horas.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Queimadura 2º grau",
        "descricao_clinica": "Queimadura com bolhas na pele, área vermelha, muito dolorosa e úmida. Pode ser por água quente, fogo, óleo ou produto químico. Área extensa ou em face, mãos e genitais é mais grave",
        "perguntas_chave": [
            "Qual foi a causa da queimadura?",
            "Têm bolhas na área queimada?",
            "A queimadura é no rosto, pescoço, mãos, pés ou genitais?",
            "Qual o tamanho aproximado da área queimada?"
        ],
        "conduta_sugerida": "Lavar com água fria corrente por 10-20 minutos. NÃO usar pasta de dente, manteiga ou qualquer produto. Cobrir com pano limpo. Encaminhar para UBS. Se face, mãos, genitais ou área maior que a palma da mão: emergência.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Afogamento com recuperação",
        "descricao_clinica": "Pessoa que quase se afogou e foi resgatada, pode parecer bem, mas tosse muito, está confusa, com respiração rápida, ou teve perda de consciência na água",
        "perguntas_chave": [
            "A pessoa ficou inconsciente ou engoliu muita água?",
            "Está com tosse intensa ou falta de ar após o resgate?",
            "Está confusa ou agitada de forma diferente do normal?",
            "Quanto tempo ficou submersa ou em dificuldade na água?"
        ],
        "conduta_sugerida": "EMERGÊNCIA mesmo que pareça bem. Afogamento secundário pode ocorrer horas depois. Encaminhar para hospital para observação mínima de 6 horas. Não liberar para casa sem avaliação médica.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Corpo estranho em criança",
        "descricao_clinica": "Criança que engoliu ou colocou objeto no nariz, ouvido ou garganta. Pode estar engasgada, com tosse repentina, dificuldade para respirar, salivando muito ou com objeto visível",
        "perguntas_chave": [
            "Viu ou suspeita que a criança engoliu ou aspirou algum objeto?",
            "Está com tosse súbita intensa ou dificuldade para respirar?",
            "Consegue engolir, falar ou chorar normalmente?",
            "Que objeto pode ter sido: moeda, pilha, brinquedo pequeno?"
        ],
        "conduta_sugerida": "Se engasgada e não consegue respirar: manobra de Heimlich imediatamente. PILHA DE BOTÃO engolida: EMERGÊNCIA — queima o esôfago em horas. Qualquer objeto aspirado: emergência. Não tentar remover com o dedo às cegas.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "crianca",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Choque anafilático",
        "descricao_clinica": "Logo após picada de inseto, alimento, remédio ou vacina: urticária intensa, inchaço na garganta, falta de ar, queda de pressão, tontura extrema, desmaio",
        "perguntas_chave": [
            "Teve contato com algo que pode ter causado alergia (comida, remédio, picada)?",
            "Há quanto tempo começaram os sintomas após o contato?",
            "Tem inchaço na garganta, lábios ou língua?",
            "Está com dificuldade para respirar ou quase desmaindo?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Ligar 192 (SAMU). Deitar com pernas elevadas. Se a pessoa tiver epinefrina autoinjetável: aplicar. Não dar nada pela boca. Cada minuto conta.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Hipoglicemia grave",
        "descricao_clinica": "Diabético que usa insulina ou remédio, tremendo muito, suando frio, confuso, com fome intensa, podendo perder a consciência. Açúcar no sangue muito baixo",
        "perguntas_chave": [
            "Tem diabetes e usa insulina ou comprimido para baixar o açúcar?",
            "Está tremendo, suando frio e com fome súbita intensa?",
            "Comeu normalmente antes de tomar o remédio ou insulina?",
            "Está conseguindo engolir ou está muito confuso?"
        ],
        "conduta_sugerida": "Se consciente e consegue engolir: dar açúcar, suco ou refrigerante comum imediatamente. Repetir em 15 minutos se não melhorar. Se inconsciente: EMERGÊNCIA — não dar nada pela boca. Ligar 192.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Intoxicação medicamentosa",
        "descricao_clinica": "Pessoa que tomou quantidade excessiva de remédio, seja acidentalmente ou intencionalmente, com sonolência extrema, confusão, vômito, respiração lenta ou irregular",
        "perguntas_chave": [
            "Que remédio e quanto foi tomado?",
            "Foi acidente ou intencional?",
            "Há quanto tempo tomou o remédio?",
            "Está consciente e respondendo normalmente?"
        ],
        "conduta_sugerida": "EMERGÊNCIA. Ligar 192 (SAMU) e o Centro de Informações Toxicológicas (0800 722 6001). Guardar a embalagem do remédio. Se intencional: não deixar sozinho e acionar suporte de saúde mental urgente.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Parada cardiorrespiratória (reconhecimento)",
        "descricao_clinica": "Pessoa desmaiada, sem resposta a chamados, sem respiração ou respirando de forma anormal (tipo ronco), sem pulso. Pode ter ocorrido após dor no peito ou sem aviso",
        "perguntas_chave": [
            "A pessoa está respondendo quando você chama pelo nome e toca no ombro?",
            "Está respirando normalmente?",
            "Houve dor no peito ou desmaio antes de cair?",
            "Há quanto tempo está assim?"
        ],
        "conduta_sugerida": "EMERGÊNCIA IMEDIATA. Ligar 192 (SAMU) agora. Iniciar RCP: 30 compressões fortes no centro do peito + 2 respirações — repetir até o socorro chegar. Cada minuto sem RCP reduz a chance de sobrevida em 10%.",
        "urgencia": 3,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Infecções comuns (8) ────────────────────────────────────────────────
    {
        "nome": "Celulite infecciosa",
        "descricao_clinica": "Área da pele vermelha, quente, inchada e dolorida que está crescendo, geralmente depois de uma ferida, arranhão ou picada de inseto. Pode ter febre",
        "perguntas_chave": [
            "A área vermelha está crescendo nas últimas horas ou dias?",
            "Está quente, dura e muito dolorida ao toque?",
            "Houve uma ferida, picada ou arranhão antes de aparecer?",
            "Tem febre ou linhas vermelhas se espalhando a partir do local?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para antibiótico. Se houver linhas vermelhas se espalhando (linfangite), febre alta ou face afetada: emergência. Marcar a borda da vermelhidão para monitorar expansão.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Abscesso cutâneo",
        "descricao_clinica": "Caroço dolorido debaixo da pele cheio de pus, quente e vermelho, que pode ter uma cabeça branca ou amarela no centro. Pode ter febre e mal-estar",
        "perguntas_chave": [
            "O caroço está aumentando de tamanho?",
            "Está muito dolorido e com calor local?",
            "Está com febre junto?",
            "Tem diabetes ou usa corticoide? (piora o risco)"
        ],
        "conduta_sugerida": "Encaminhar para UBS para drenagem por profissional. NÃO espremer em casa — risco de disseminar infecção. Se diabético ou abscesso no rosto: maior prioridade. Compressas mornas ajudam a amadurecer.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Conjuntivite infecciosa",
        "descricao_clinica": "Olho vermelho com secreção amarela ou verde que gruda as pálpebras, lacrimejamento, sensação de areia no olho. Muito contagiosa. Pode afetar um ou dois olhos",
        "perguntas_chave": [
            "Os olhos estão vermelhos com secreção amarela ou verde?",
            "A pálpebra está grudada de manhã ao acordar?",
            "Outras pessoas próximas com o mesmo problema?",
            "Tem dor intensa, visão embaçada ou sensibilidade à luz?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para colírio antibiótico se bacteriana. Orientar lavagem das mãos frequente, não compartilhar toalhas. Se dor intensa ou perda de visão: emergência oftalmológica.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Otite média aguda",
        "descricao_clinica": "Criança (ou adulto) com dor de ouvido intensa, febre, pode ter saída de líquido pelo ouvido. Criança pequena fica muito irritada, puxando o ouvido e com choro intenso",
        "perguntas_chave": [
            "Está com dor intensa no ouvido?",
            "Saiu líquido ou pus pelo ouvido?",
            "Está com febre junto?",
            "Teve resfriado ou infecção respiratória recentemente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para avaliação e antibiótico se necessário. Não colocar nada no ouvido. Se saída de pus, febre alta ou criança muito pequeña: prioridade na UBS.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Faringite estreptocócica",
        "descricao_clinica": "Dor de garganta intensa de início súbito, febre alta, garganta muito vermelha com pontos brancos, dificuldade para engolir, sem tosse ou coriza",
        "perguntas_chave": [
            "A dor de garganta começou de repente com febre alta?",
            "Tem pontos brancos na garganta?",
            "Não tem tosse nem coriza?",
            "Tem gânglios (íngua) doloridos no pescoço?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para swab ou teste rápido. Tratar com antibiótico por 10 dias completos para evitar febre reumática. Não interromper o antibiótico mesmo com melhora.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Infecção urinária",
        "descricao_clinica": "Dor ou ardência ao urinar, urina turva com cheiro forte, vontade frequente de urinar com pouca saída, dor na parte baixa da barriga. Em idosos pode causar confusão mental",
        "perguntas_chave": [
            "Sente dor ou ardência ao urinar?",
            "A urina está turva, escura ou com cheiro diferente?",
            "Tem dor nas costas (altura dos rins) ou febre?",
            "Em idosos: está mais confuso que o normal recentemente?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para exame de urina e antibiótico. Se febre, calafrios e dor nas costas (pielonefrite): encaminhar com urgência. Orientar hidratação abundante.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Candidíase vaginal",
        "descricao_clinica": "Mulher com coceira intensa na vagina, corrimento branco em grumos parecido com queijo cottage, ardência ao urinar, vermelhidão e inchaço na região genital",
        "perguntas_chave": [
            "Tem coceira intensa na vagina ou na vulva?",
            "O corrimento é branco, grumoso e sem cheiro forte?",
            "Usou antibiótico recentemente ou tem diabetes?",
            "Já teve esse problema antes? Usou algum tratamento?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para confirmação e tratamento com antifúngico. Orientar higiene adequada sem duchas vaginais. Se recorrente ou gestante: avaliação médica necessária.",
        "urgencia": 1,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Herpes zoster (cobreiro)",
        "descricao_clinica": "Dor intensa seguida de bolhas que seguem uma faixa de um lado do corpo ou do rosto, como uma linha. Mais comum em idosos ou pessoas com imunidade baixa. Muito doloroso",
        "perguntas_chave": [
            "A dor e as bolhas estão dispostas em faixa de um lado do corpo?",
            "A dor começou antes das bolhas aparecerem?",
            "Tem mais de 60 anos ou está com imunidade baixa?",
            "As bolhas estão no rosto perto dos olhos?"
        ],
        "conduta_sugerida": "Encaminhar para UBS com urgência — o antiviral é mais eficaz nas primeiras 72 horas. Se bolhas próximas aos olhos: emergência oftalmológica. Não é urgência se for leve e longe dos olhos.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "idoso",
        "fonte": "MS-CAB",
        "validado_por": None,
    },

    # ── Situações especiais (4) ─────────────────────────────────────────────
    {
        "nome": "Violência doméstica suspeita",
        "descricao_clinica": "Mulher, criança ou idoso com machucados em locais incomuns, marcas de mordida, queimaduras de cigarro, história que não bate com o machucado, ou que demonstra medo ao falar sobre como se machucou",
        "perguntas_chave": [
            "Como aconteceu esse machucado? A história faz sentido?",
            "Os machucados estão em diferentes fases de cicatrização?",
            "A pessoa parece com medo ao responder ou quando o acompanhante está perto?",
            "Já houve outras situações parecidas antes?"
        ],
        "conduta_sugerida": "Criar oportunidade para conversar a sós com a pessoa. Não confrontar o agressor. Registrar as lesões com descrição detalhada. Acionar CRAS, CREAS ou delegacia da mulher conforme o caso. Notificar à vigilância epidemiológica (ficha de violência). Não expor a vítima a risco.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
    {
        "nome": "Desnutrição em adulto",
        "descricao_clinica": "Adulto ou idoso muito emagrecido sem estar em dieta, com perda de mais de 10% do peso em poucos meses, fraqueza muscular intensa, feridas que não cicatrizam, cabelo caindo",
        "perguntas_chave": [
            "Quanto perdeu de peso e em quanto tempo?",
            "Está conseguindo comer normalmente ou tem dificuldade?",
            "Tem doença crônica que possa estar causando o emagrecimento?",
            "Está em situação de insegurança alimentar (sem dinheiro para comida)?"
        ],
        "conduta_sugerida": "Encaminhar para UBS para investigação da causa. Registrar no SISVAN. Acionar CRAS se houver insegurança alimentar. Se idoso acamado com escaras: encaminhar com urgência.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "adulto",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Síncope (desmaio)",
        "descricao_clinica": "Pessoa que desmaiou por alguns segundos a minutos, com recuperação completa rápida. Pode ter sido precedido por tontura, visão escurecendo e palidez",
        "perguntas_chave": [
            "Perdeu a consciência completamente? Por quanto tempo?",
            "Teve tontura ou visão escurecendo antes de desmaiar?",
            "Estava em pé há muito tempo, com calor ou após susto?",
            "Tem histórico de problema no coração?"
        ],
        "conduta_sugerida": "Deitar a pessoa com pernas elevadas. Se recuperação completa e sem histórico cardíaco: encaminhar para UBS. Se ocorreu durante esforço físico, com convulsão ou sem causa aparente: emergência para investigar causa cardíaca.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "MS-CAB",
        "validado_por": None,
    },
    {
        "nome": "Dor abdominal aguda suspeita",
        "descricao_clinica": "Dor forte na barriga de início súbito, que pode estar localizada em um ponto específico, com ou sem febre, náusea e vômito. Pode indicar apendicite, cálculo renal ou outros problemas sérios",
        "perguntas_chave": [
            "A dor começou de repente e está piorando?",
            "A dor está localizada em um ponto fixo da barriga?",
            "Tem febre, vômito ou a barriga está dura ao toque?",
            "A dor irradia para as costas, virilha ou ombro?"
        ],
        "conduta_sugerida": "Encaminhar para UBS ou urgência para avaliação médica. NÃO dar remédio para dor antes da avaliação (pode mascarar o diagnóstico). Se barriga dura, febre alta ou dor incapacitante: emergência.",
        "urgencia": 2,
        "regioes": ["todos"],
        "faixa_etaria": "todos",
        "fonte": "IA-pendente-validacao",
        "validado_por": None,
    },
]

# ── Inserção no banco ───────────────────────────────────────────────────────
vectors = [model.encode(c["descricao_clinica"]).tolist() for c in condicoes]

points = [
    PointStruct(id=i, vector=vectors[i], payload=condicoes[i])
    for i in range(len(condicoes))
]

client.points.upsert(COLECAO, points=points)

# ── Resumo final ────────────────────────────────────────────────────────────
contagem_fonte = Counter(c["fonte"] for c in condicoes)
validadas = sum(1 for c in condicoes if c["validado_por"] is not None)

print(f"\nTotal inserido: {len(condicoes)} condições")
print(f"  - MS-CAB: {contagem_fonte.get('MS-CAB', 0)} condições")
print(f"  - MS-GVS: {contagem_fonte.get('MS-GVS', 0)} condições")
print(f"  - IA-pendente-validacao: {contagem_fonte.get('IA-pendente-validacao', 0)} condições")
print(f"  - Validadas por médico: {validadas} condições")
