import cv2  # Importa a biblioteca OpenCV para processamento de imagens e vídeo
import numpy as np  # Importa a biblioteca NumPy para operações numéricas
import os  # Importa a biblioteca OS para interações com o sistema operacional
import speech_recognition as sr  # Importa a biblioteca SpeechRecognition para reconhecimento de voz
import pyttsx3  # Importa a biblioteca pyttsx3 para síntese de fala
import requests  # Importa a biblioteca Requests para fazer requisições HTTP
import json  # Importa a biblioteca JSON para manipulação de dados JSON
import time  # Importa a biblioteca Time para manipulações relacionadas ao tempo
import sys  # Importa a biblioteca Sys para interações com o sistema (necessário para encerrar o programa)
from datetime import datetime  # Importa a classe datetime para manipulação de datas e horas
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importa ImageDataGenerator do Keras para aumento de dados
from keras.models import Sequential  # Importa o modelo sequencial do Keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D  # Importa camadas do Keras para construir redes neurais
from keras.optimizers import Adam  # Importa o otimizador Adam do Keras
import keras.utils as utils  # Importa utilitários do Keras
import easyocr  # Importa a biblioteca EasyOCR para reconhecimento óptico de caracteres
import cvlib as cv  # Importa a biblioteca CVLib para detecção de objetos
from cvlib.object_detection import draw_bbox  # Importa a função draw_bbox do CVLib para desenhar caixas delimitadoras
import tensorflow as tf  # Importa a biblioteca TensorFlow

# Configurações da captura de faces e treinamento
class FaceRecognition:
    def __init__(self, dir_fotos='fotos/', largura=220, altura=220, num_amostras=30):
        self.dir_fotos = dir_fotos  # Define o diretório para salvar as fotos
        self.largura = largura  # Define a largura das imagens das faces
        self.altura = altura  # Define a altura das imagens das faces
        self.num_amostras = num_amostras  # Define o número de amostras a serem capturadas
        self.classificador = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Carrega o classificador Haar para detecção de faces
        self.eigen = cv2.face.EigenFaceRecognizer_create()  # Cria um reconhecedor de faces EigenFace
        self.camera = cv2.VideoCapture(0)  # Inicializa a captura de vídeo da câmera padrão

        if not os.path.exists(self.dir_fotos):
            os.makedirs(self.dir_fotos)  # Cria o diretório de fotos se não existir

    def capturar_faces(self, nome):
        amostra = 1  # Inicializa o contador de amostras
        print(f"Capturando imagens para {nome}. Pressione 'c' para capturar uma imagem e 'q' para sair.")  # Informa o usuário
        while True:
            status, imagem = self.camera.read()  # Lê um frame da câmera
            if not status:
                print("Erro ao acessar a câmera.")  # Informa erro caso não consiga acessar a câmera
                break

            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza
            facesDetectadas = self.classificador.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))  # Detecta faces na imagem

            for (x, y, w, h) in facesDetectadas:
                cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenha retângulos nas faces detectadas

            cv2.imshow("Face detectada", imagem)  # Exibe a imagem com as faces detectadas

            key = cv2.waitKey(1) & 0xFF  # Captura a tecla pressionada
            if key == ord('c') and len(facesDetectadas) > 0:
                for (x, y, w, h) in facesDetectadas:
                    imagemFace = cv2.resize(imagemCinza[y:y + h, x:x + w], (self.largura, self.altura))  # Redimensiona a face detectada
                    localFoto = os.path.join(self.dir_fotos, f'{nome}_{amostra}.jpg')  # Define o caminho para salvar a foto
                    cv2.imwrite(localFoto, imagemFace)  # Salva a foto da face
                    print(f"Imagem {amostra} capturada e salva como {localFoto}")  # Informa que a imagem foi salva
                    amostra += 1  # Incrementa o contador de amostras

            if amostra > self.num_amostras:
                print(f"Capturas finalizadas após {self.num_amostras} imagens para {nome}.")  # Informa que a captura foi finalizada
                break

            if key == ord('q'):
                print("Encerrado manualmente.")  # Informa que a captura foi encerrada manualmente
                break

        cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

    def treinar_modelo(self):
        caminhos = [os.path.join(self.dir_fotos, f) for f in os.listdir(self.dir_fotos)]  # Obtém os caminhos de todas as fotos
        faces = []  # Lista para armazenar as faces
        nomes = []  # Lista para armazenar os nomes correspondentes

        for caminhoImagem in caminhos:
            imagemFace = cv2.imread(caminhoImagem)  # Lê a imagem da face
            imagemCinza = cv2.cvtColor(imagemFace, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza
            nome = os.path.split(caminhoImagem)[-1].split('_')[0]  # Extrai o nome da imagem

            nomes.append(nome)  # Adiciona o nome à lista de nomes
            faces.append(imagemCinza)  # Adiciona a face à lista de faces

        nomes_unicos = list(set(nomes))  # Obtém os nomes únicos
        nome_indices = np.array([nomes_unicos.index(nome) for nome in nomes])  # Cria um array de índices para os nomes

        self.eigen.train(faces, nome_indices)  # Treina o reconhecedor EigenFace com as faces e índices
        self.eigen.write('classificadoreigen.yml')  # Salva o modelo treinado em um arquivo

    def reconhecer_face(self):
        detectorFace = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Carrega o classificador Haar para detecção de faces
        reconhecedor = cv2.face.EigenFaceRecognizer_create()  # Cria um reconhecedor de faces EigenFace
        reconhecedor.read('classificadoreigen.yml')  # Carrega o modelo treinado

        nomes_unicos = list(set([os.path.split(f)[-1].split('_')[0] for f in os.listdir(self.dir_fotos)]))  # Obtém os nomes únicos das fotos

        camera = cv2.VideoCapture(0)  # Inicializa a captura de vídeo da câmera
        nomeReconhecido = None  # Inicializa a variável para armazenar o nome reconhecido
        reconhecido = False  # Flag para indicar se a face foi reconhecida
        while True:
            status, imagem = camera.read()  # Lê um frame da câmera
            if not status:
                print("Erro ao acessar a câmera.")  # Informa erro caso não consiga acessar a câmera
                break

            imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza
            facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))  # Detecta faces na imagem

            for x, y, l, a in facesDetectadas:
                imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (self.altura, self.largura))  # Redimensiona a face detectada
                cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)  # Desenha um retângulo na face detectada
                indice_nome, confianca = reconhecedor.predict(imagemFace)  # Reconhece a face e obtém a confiança

                if indice_nome < len(nomes_unicos):
                    nomeReconhecido = nomes_unicos[indice_nome]  # Obtém o nome reconhecido
                else:
                    nomeReconhecido = "Desconhecido"  # Define como desconhecido se o índice não for válido

                if confianca < 5000:
                    print(f"Bem vindo {nomeReconhecido}")  # Saudação ao usuário reconhecido
                    cv2.putText(imagem, nomeReconhecido, (x, y + a + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Exibe o nome reconhecido na imagem
                    reconhecido = True  # Atualiza a flag de reconhecimento

            cv2.imshow("Reconhecimento Facial", imagem)  # Exibe a imagem com o reconhecimento

            if reconhecido:
                cv2.waitKey(3000)  # Espera 3 segundos
                break

            if cv2.waitKey(1) == ord('q'):
                break  # Sai do loop se 'q' for pressionado

        camera.release()  # Libera a câmera
        cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

        if reconhecido:
            return nomeReconhecido  # Retorna o nome reconhecido
        return None  # Retorna None se não reconhecido

# Assistente virtual
class AssistenteDuda:
    def __init__(self, agenda_file, weather_api_key):
        self.agenda_file = agenda_file  # Define o caminho do arquivo de agenda
        self.weather_api_key = weather_api_key  # Define a chave da API do OpenWeatherMap
        self.duda = pyttsx3.init()  # Inicializa o mecanismo de síntese de fala
        self.duda.setProperty('rate', 162)  # Define a taxa de fala
        self.duda.setProperty('volume', 2.0)  # Define o volume da fala
        voices = self.duda.getProperty('voices')  # Obtém as vozes disponíveis
        # Selecionar voz feminina em português do Brasil
        for voice in voices:
            if "Maria" in voice.name:
                self.duda.setProperty('voice', voice.id)  # Seleciona a voz "Maria" se disponível
                break
        self.reconhecedor = sr.Recognizer()  # Inicializa o reconhecedor de voz
        self.reader = easyocr.Reader(['pt'])  # Inicializa o leitor EasyOCR para português

    def falar_e_escrever(self, texto):
        """Função que fala e também escreve o texto no terminal"""
        print(texto)  # Exibe o texto no terminal
        self.duda.say(texto)  # Fala o texto
        self.duda.runAndWait()  # Executa a fala

    def salvar_arquivo(self, evento):
        with open(self.agenda_file, "a", encoding="utf-8") as file:
            file.write(evento + "\n")  # Salva o evento no arquivo de agenda

    def ler_agenda(self):
        try:
            with open(self.agenda_file, "r", encoding="utf-8") as file:
                return file.read()  # Retorna o conteúdo da agenda
        except FileNotFoundError:
            return "Nenhum evento encontrado."  # Retorna mensagem se o arquivo não for encontrado

    def ouvir_comando(self):
        with sr.Microphone() as mic:
            self.reconhecedor.adjust_for_ambient_noise(mic, duration=2)  # Ajusta para ruído ambiente
            print("Aguardando comando de voz...")  # Informa que está aguardando comando
            try:
                audio = self.reconhecedor.listen(mic, timeout=10)  # Escuta o áudio do microfone
            except sr.WaitTimeoutError:
                self.falar_e_escrever("Tempo esgotado. Por favor, tente novamente.")  # Informa timeout
                return ""
            try:
                comando = self.reconhecedor.recognize_google(audio, language='pt-BR')  # Reconhece o comando de voz
                print(f"Comando reconhecido: {comando}")  # Exibe o comando reconhecido
                return comando.lower()  # Retorna o comando em minúsculas
            except sr.UnknownValueError:
                print("Não entendi o comando.")  # Informa que não entendeu o comando
                return ""
            except sr.RequestError:
                self.falar_e_escrever("Não consegui acessar o serviço de reconhecimento de voz.")  # Informa erro de serviço
                return ""

    def obter_temperatura(self, cidade):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={self.weather_api_key}&units=metric&lang=pt_br"  # Define a URL da API
        try:
            response = requests.get(url)  # Faz a requisição HTTP
            response.raise_for_status()  # Levanta exceção para códigos de status de erro
            dados = response.json()  # Obtém os dados em formato JSON

            temperatura = dados['main']['temp']  # Obtém a temperatura
            descricao = dados['weather'][0]['description']  # Obtém a descrição do clima
            resposta = f"A temperatura em {cidade} é de {temperatura} graus Celsius com {descricao}."  # Formata a resposta
        except requests.exceptions.HTTPError as http_err:
            resposta = f"Erro na requisição HTTP: {http_err}"  # Informa erro HTTP
        except Exception as err:
            resposta = f"Ocorreu um erro: {err}"  # Informa erro genérico

        return resposta  # Retorna a resposta para ser utilizada na função responder_pergunta

    def obter_hora_atual(self):
        agora = datetime.now()  # Obtém o momento atual
        hora = agora.strftime("%H:%M")  # Formata a hora
        resposta = f"São {hora} agora."  # Formata a resposta
        return resposta  # Retorna a resposta para ser utilizada na função responder_pergunta

    def responder_pergunta(self, pergunta):
        if "temperatura" in pergunta:
            cidade = "São Paulo"  # Define a cidade como São Paulo para facilitar o processamento
            resposta = self.obter_temperatura(cidade)  # Obtém a resposta da função de temperatura
        elif "hora" in pergunta:
            resposta = self.obter_hora_atual()  # Obtém a resposta da função de hora
        else:
            # Integração com a IA Generativa LLaMA
            url = "http://localhost:11434/api/generate"  # URL da API do LLaMA
            input_json = {"model": "llama3.1", "prompt": pergunta}  # Dados para requisição
            inicio = time.time()  # Início do tempo
            try:
                response = requests.post(url, json=input_json)  # Faz a requisição POST
                response.raise_for_status()  # Levanta exceção para status codes de erro
            except requests.exceptions.RequestException as e:
                self.falar_e_escrever(f"Erro ao se comunicar com a API LLaMA: {e}")  # Informa erro na comunicação
                return

            linhas = response.text.strip().split('\n')  # Divide a resposta em linhas
            valores_response = [json.loads(linha).get('response') for linha in linhas if linha.strip()]  # Extrai as respostas
            resposta_completa = ''.join(valores_response)  # Concatena as respostas

            # Truncar a resposta para 200 caracteres com reticências se necessário
            resposta = resposta_completa[:200] + '...' if len(resposta_completa) > 200 else resposta_completa  # Trunca a resposta

            print("Tempo: ", time.time() - inicio)  # Exibe o tempo de resposta

        print(f"Resposta: {resposta}")  # Exibe a resposta no prompt
        self.duda.say(resposta)  # Fala a resposta
        self.duda.runAndWait()  # Executa a fala

    def detectar_objetos_cvlib(self):
        """Função que abre a câmera e realiza detecção de objetos usando cvlib."""
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Inicializa a captura de vídeo da câmera com DirectShow

            if not cap.isOpened():
                print("Erro: Não foi possível acessar a webcam.")  # Informa erro se não conseguir acessar a webcam
                self.falar_e_escrever("Erro ao acessar a webcam.")  # Fala o erro
                return

            objetos_detectados = []  # Lista para armazenar objetos detectados

            while True:
                ret, frame = cap.read()  # Lê um frame da câmera

                if not ret:
                    print("Erro ao capturar o frame da webcam.")  # Informa erro ao capturar o frame
                    self.falar_e_escrever("Erro ao capturar o frame da webcam.")  # Fala o erro
                    break

                bbox, label, conf = cv.detect_common_objects(frame, confidence=0.50, model='yolov3-tiny')  # Detecta objetos no frame

                objetos_detectados.extend(label)  # Adiciona os objetos detectados à lista

                print(f"Objetos detectados: {label}, Confiança: {conf}")  # Exibe os objetos detectados e suas confianças

                out = draw_bbox(frame, bbox, label, conf, write_conf=True)  # Desenha as caixas delimitadoras nos objetos

                cv2.imshow("Detecção de Objetos em Tempo Real", out)  # Exibe o frame com os objetos detectados

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # Sai do loop se 'q' for pressionado

            if objetos_detectados:
                objetos_unicos = set(objetos_detectados)  # Obtém objetos únicos
                objetos_str = ", ".join(objetos_unicos)  # Concatena os objetos em uma string
                self.falar_e_escrever(f"Os seguintes objetos foram detectados: {objetos_str}")  # Informa os objetos detectados
            else:
                self.falar_e_escrever("Nenhum objeto foi detectado.")  # Informa que nenhum objeto foi detectado

            self.falar_e_escrever("Detecção de objetos encerrada.")  # Informa que a detecção foi encerrada
        except Exception as e:
            print(f"Ocorreu um erro: {e}")  # Exibe o erro ocorrido
            self.falar_e_escrever("Ocorreu um erro durante a detecção de objetos.")  # Fala o erro
        finally:
            cap.release()  # Libera a câmera
            cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

        # Perguntar se deseja algo mais
        self.perguntar_deseja_algo_mais()  # Chama a função para perguntar se o usuário deseja algo mais

    def reconhecer_escrita(self):
        camera = cv2.VideoCapture(0)  # Inicializa a captura de vídeo da câmera
        print("Câmera aberta. Aguardando reconhecimento de texto...")  # Informa que a câmera está aberta
        while True:
            status, frame = camera.read()  # Lê um frame da câmera
            if not status:
                print("Erro ao acessar a câmera.")  # Informa erro caso não consiga acessar a câmera
                break

            cv2.imshow("Reconhecendo Escrita", frame)  # Exibe o frame atual

            resultados = self.reader.readtext(frame)  # Realiza o reconhecimento de texto no frame
            texto = " ".join([resultado[1] for resultado in resultados])  # Concatena os textos reconhecidos

            if len(texto.strip()) > 0:
                print("Texto detectado! Fechando a câmera...")  # Informa que o texto foi detectado
                cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV
                break

            if cv2.waitKey(1) == ord('q'):
                break  # Sai do loop se 'q' for pressionado

        camera.release()  # Libera a câmera
        time.sleep(2)  # Aguarda 2 segundos
        self.falar_e_escrever(f"O texto reconhecido é: {texto}")  # Fala o texto reconhecido

        # Perguntar se deseja reconhecer outra escrita
        self.perguntar_deseja_reconhecer_outra_escrita()  # Chama a função para perguntar se deseja reconhecer outra escrita

    # Função para treinar objetos com Keras
    def treinar_objetos_keras(self):
        cap = cv2.VideoCapture(0)  # Inicializa a captura de vídeo da câmera
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtém a largura do frame
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtém a altura do frame
        print("Resolution:", width, height)  # Exibe a resolução da câmera

        model = tf.keras.Sequential([  # Define o modelo sequencial do TensorFlow Keras
            tf.keras.layers.Input(shape=(224, 224, 3)),  # Camada de entrada com forma 224x224x3
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # Camada convolucional com 32 filtros
            tf.keras.layers.MaxPooling2D(2, 2),  # Camada de pooling
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Outra camada convolucional com 64 filtros
            tf.keras.layers.MaxPooling2D(2, 2),  # Outra camada de pooling
            tf.keras.layers.Flatten(),  # Achata os dados para a camada densa
            tf.keras.layers.Dense(128, activation='relu'),  # Camada densa com 128 neurônios
            tf.keras.layers.Dense(3, activation='softmax', name='output')  # Camada de saída com 3 classes
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),  # Compila o modelo com otimizador Adam
                      loss='categorical_crossentropy',  # Define a função de perda
                      metrics=['accuracy'])  # Define as métricas de avaliação

        images = []  # Lista para armazenar as imagens capturadas
        y = []  # Lista para armazenar os rótulos das classes
        labels = ["Caneca", "Carta", "Aviao"]  # Define os rótulos das classes
        EPOCHS = 10  # Define o número de épocas para treinamento
        BS = 8  # Define o tamanho do lote (batch size)

        print("Pressione '1', '2', ou '3' para capturar imagens das classes correspondentes.")  # Instruções para captura
        print("Pressione 't' para treinar o modelo após coletar as imagens.")  # Instruções para treinamento
        print("Pressione 'p' para iniciar as predições em tempo real.")  # Instruções para predição
        print("Pressione 'q' para sair.")  # Instruções para sair

        while True:
            ret, frame = cap.read()  # Lê um frame da câmera
            if not ret:
                print("Erro ao capturar o frame da webcam.")  # Informa erro ao capturar o frame
                break

            cv2.imshow("Video", frame)  # Exibe o frame atual
            k = cv2.waitKey(1) & 0xff  # Captura a tecla pressionada

            if k == ord('q'):
                break  # Sai do loop se 'q' for pressionado
            elif k == ord('1'):
                images.append(frame.copy())  # Adiciona a imagem à lista de imagens
                y.append([1.0, 0.0, 0.0])  # Adiciona o rótulo da classe 1
                print("Imagem capturada para a classe 1")  # Informa que a imagem foi capturada para a classe 1
            elif k == ord('2'):
                images.append(frame.copy())  # Adiciona a imagem à lista de imagens
                y.append([0.0, 1.0, 0.0])  # Adiciona o rótulo da classe 2
                print("Imagem capturada para a classe 2")  # Informa que a imagem foi capturada para a classe 2
            elif k == ord('3'):
                images.append(frame.copy())  # Adiciona a imagem à lista de imagens
                y.append([0.0, 0.0, 1.0])  # Adiciona o rótulo da classe 3
                print("Imagem capturada para a classe 3")  # Informa que a imagem foi capturada para a classe 3
            elif k == ord('t') and len(images) > 0:
                print("Treinando o modelo...")  # Informa que o treinamento está começando
                images_np = np.array([cv2.resize(img, (224, 224)) for img in images])  # Redimensiona todas as imagens
                images_np = images_np / 255.0  # Normaliza as imagens
                y_np = np.array(y)  # Converte os rótulos para um array NumPy
                model.fit(images_np, y_np, epochs=EPOCHS, batch_size=BS, validation_split=0.2)  # Treina o modelo
                print("Treinamento concluído")  # Informa que o treinamento foi concluído
            elif k == ord('p'):
                if len(images) == 0:
                    print("Por favor, treine o modelo antes de iniciar as predições.")  # Informa que é necessário treinar antes
                    continue
                print("Iniciando predições em tempo real...")  # Informa que as predições estão iniciando
                while True:
                    ret_pred, frame_pred = cap.read()  # Lê um frame da câmera
                    if not ret_pred:
                        break
                    frame_resized = cv2.resize(frame_pred, (224, 224))  # Redimensiona o frame para 224x224
                    frame_normalized = frame_resized / 255.0  # Normaliza o frame
                    frame_input = np.expand_dims(frame_normalized, axis=0)  # Expande as dimensões para o modelo

                    predict = model.predict(frame_input)  # Faz a predição
                    predict_label = np.argmax(predict[0])  # Obtém o índice da classe com maior probabilidade
                    label_text = labels[predict_label]  # Obtém o rótulo da classe

                    cv2.putText(frame_pred, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Escreve o rótulo na imagem
                    cv2.imshow("Predição em Tempo Real", frame_pred)  # Exibe a predição em tempo real

                    k_pred = cv2.waitKey(1) & 0xFF  # Captura a tecla pressionada
                    if k_pred == ord('s') or k_pred == ord('q'):
                        print("Parando predições em tempo real")  # Informa que as predições estão parando
                        break

                print("Voltando ao modo de captura")  # Informa que está voltando ao modo de captura

        cap.release()  # Libera a câmera
        cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

        # Perguntar se deseja algo mais
        self.perguntar_deseja_algo_mais()  # Chama a função para perguntar se o usuário deseja algo mais

    def perguntar_deseja_algo_mais(self):
        while True:
            self.falar_e_escrever("Deseja algo mais?")  # Pergunta se o usuário deseja algo mais
            resposta = self.ouvir_comando()  # Ouve a resposta do usuário
            if "não" in resposta:
                self.falar_e_escrever("Até mais!")  # Despede-se do usuário
                sys.exit()  # Encerra o programa
            elif "sim" in resposta:
                self.falar_e_escrever("O que mais deseja saber?")  # Pergunta o que mais o usuário deseja
                break  # Sai do loop e volta a aguardar comando
            else:
                self.falar_e_escrever("Desculpe, não entendi. Por favor, diga 'sim' ou 'não'.")  # Informa que não entendeu

    def perguntar_deseja_reconhecer_outra_escrita(self):
        while True:
            self.falar_e_escrever("Deseja reconhecer outra escrita?")  # Pergunta se o usuário deseja reconhecer outra escrita
            resposta = self.ouvir_comando()  # Ouve a resposta do usuário
            if "não" in resposta:
                self.falar_e_escrever("Voltando ao menu principal.")  # Informa que está voltando ao menu principal
                break  # Sai do loop e volta ao menu principal
            elif "sim" in resposta:
                self.reconhecer_escrita()  # Chama a função de reconhecimento de escrita novamente
                break  # Sai do loop
            else:
                self.falar_e_escrever("Desculpe, não entendi. Por favor, diga 'sim' ou 'não'.")  # Informa que não entendeu

    def executar(self):
        while True:
            self.falar_e_escrever("Diga Ok Duda para começar.")  # Informa para dizer "Ok Duda" para começar
            comando = self.ouvir_comando()  # Ouve o comando do usuário
            if "ok duda" in comando:
                self.falar_e_escrever("Diga Mestre, como posso ajudá-lo?")  # Informa que está pronta para ajudar
                while True:
                    comando = self.ouvir_comando()  # Ouve o comando do usuário
                    if comando:
                        print(f"Você disse: {comando}")  # Exibe o comando no terminal
                        if "agendar evento" in comando:
                            self.falar_e_escrever("Ok, qual evento devo cadastrar?")  # Pergunta qual evento cadastrar
                            evento = self.ouvir_comando()  # Ouve o evento
                            if evento:
                                self.salvar_arquivo(evento)  # Salva o evento no arquivo de agenda
                                self.falar_e_escrever(f"Evento '{evento}' cadastrado com sucesso!")  # Informa que o evento foi cadastrado
                            else:
                                self.falar_e_escrever("Não consegui entender o evento.")  # Informa que não entendeu o evento
                        elif "ler agenda" in comando:
                            conteudo_agenda = self.ler_agenda()  # Lê o conteúdo da agenda
                            self.falar_e_escrever("Aqui está a sua agenda.")  # Informa que está mostrando a agenda
                            self.falar_e_escrever(conteudo_agenda)  # Fala o conteúdo da agenda
                        elif "reconhecer escrita" in comando:
                            self.reconhecer_escrita()  # Chama a função para reconhecer escrita
                        elif "reconhecer objetos" in comando:
                            self.detectar_objetos_cvlib()  # Chama a função para detectar objetos
                        elif "treinar objetos" in comando:
                            self.treinar_objetos_keras()  # Chama a função para treinar objetos com Keras
                        elif "temperatura" in comando:
                            self.falar_e_escrever("Qual cidade deseja saber a temperatura?")  # Pergunta a cidade para a temperatura
                            cidade = self.ouvir_comando()  # Ouve a cidade
                            if cidade:
                                resposta_temp = self.obter_temperatura(cidade)  # Obtém a temperatura da cidade
                                self.falar_e_escrever(resposta_temp)  # Fala a temperatura
                            else:
                                self.falar_e_escrever("Não consegui entender a cidade.")  # Informa que não entendeu a cidade
                        elif "hora" in comando:
                            resposta_hora = self.obter_hora_atual()  # Obtém a hora atual
                            self.falar_e_escrever(resposta_hora)  # Fala a hora atual
                        else:
                            # Tratar como pergunta complexa usando LLaMA
                            self.responder_pergunta(comando)  # Chama a função para responder a pergunta complexa

                    # Implementação do fluxo "Deseja algo mais?"
                    self.perguntar_deseja_algo_mais()  # Pergunta se o usuário deseja algo mais

# Ponto de entrada do programa
if __name__ == "__main__":
    face_recognition = FaceRecognition()  # Inicializa a classe de reconhecimento facial

    while True:
        nome = input("Digite o nome da pessoa a ser cadastrada: ")  # Solicita o nome da pessoa
        face_recognition.capturar_faces(nome)  # Captura as faces da pessoa

        adicionar_mais = input("Deseja cadastrar outra face? (sim/nao): ").strip().lower()  # Pergunta se deseja cadastrar outra face
        if adicionar_mais == 'sim':
            continue  # Continua o loop para cadastrar outra face
        else:
            break  # Sai do loop

    face_recognition.treinar_modelo()  # Treina o modelo de reconhecimento facial

    nome_reconhecido = face_recognition.reconhecer_face()  # Reconhece a face e obtém o nome

    if nome_reconhecido:
        print(f"Bem vindo {nome_reconhecido}")  # Saudação ao usuário reconhecido
        assistente = AssistenteDuda(
            r"C:\Users\cesar\Desktop\FIAP\2-Ano\2 Semestre\Deep Learning & AI\agenda.txt",  # Define o caminho do arquivo de agenda
            "b2312a3612650c129498b6c7b10b0e8a"  # Chave de API válida do OpenWeatherMap
        )
        assistente.falar_e_escrever(f"Bem vindo, {nome_reconhecido}")  # Fala a saudação ao usuário
        assistente.executar()  # Executa o assistente virtual
