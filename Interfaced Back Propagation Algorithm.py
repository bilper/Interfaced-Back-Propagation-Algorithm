import numpy as np 
import matplotlib.pyplot as plt
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.core.fromnumeric import shape

root = tkinter.Tk()
root.geometry("900x660")
np.random.seed(42)

#bg = tkinter.PhotoImage(file = r"aaa.png") 
  
# Show image using label 
#label1 = tkinter.Label( root, image = bg) 
#label1.place(x = 0, y = 0) 

class GUI:

    def __init__(self, master):
        #Input datasets
        

        self.learningRate = tkinter.DoubleVar()
        self.learningRate.set(0.5)

        self.hiddenNeurons = tkinter.IntVar()
        self.hiddenNeurons.set(2)

        self.entryXVar = tkinter.IntVar()
        self.entryXVar.set(4)
        self.entryYVar = tkinter.IntVar()
        self.entryYVar.set(1)

        self.epoch = tkinter.IntVar()

        self.epoch.set(10000)

        # GUI part
        labelX = tkinter.Label(text='Girdi Sayısı:')
        labelX.grid(row=0, column=0)

        entryX = tkinter.Entry(textvariable=self.entryXVar)
        entryX.grid(row=0, column=1, padx=5)

        labelY = tkinter.Label(text='Çıktı Sayısı:')
        labelY.grid(row=1, column=0)

        entryY = tkinter.Entry(textvariable=self.entryYVar)
        entryY.grid(row=1, column=1, pady=5, padx=5)

        labelHiddenLayer = tkinter.Label(text='Gizli Katman Sayısı: ')
        labelHiddenLayer.grid(row=2, column=0)

        entryHiddenLayer = tkinter.Entry(textvariable=self.hiddenNeurons)
        entryHiddenLayer.grid(row=2, column=1)

        labelLearningRate = tkinter.Label(text='Öğrenme Katsayısı: ')
        labelLearningRate.grid(row=3, column=0)

        entryLearningRate = tkinter.Entry(textvariable=self.learningRate)
        entryLearningRate.grid(row=3, column=1)

        labelEpochs = tkinter.Label(text='İterasyon Sayısı:')
        labelEpochs.grid(row=4, column=0)

        entryEpochs = tkinter.Entry(textvariable=self.epoch)
        entryEpochs.grid(row=4, column=1)

        buttonTrain = tkinter.Button(text='   Eğit   ', command=self.train)
        buttonTrain.grid(row=5, column=1, pady=5)

        labelExpectedAnswer = tkinter.Label(text='Beklenen Cevap:')
        labelExpectedAnswer.grid(row=7, column=0, pady=5)

        entryExpectedAnswer = tkinter.Listbox(root)
        entryExpectedAnswer.grid(row=8, column=0)

        labelAnswer = tkinter.Label(text='Cevap:')
        labelAnswer.grid(row=7, column=1)

        entryAnswer = tkinter.Listbox(root)
        entryAnswer.grid(row=8, column=1, pady=10)

        labelOutputWeights = tkinter.Label(text='output weights:')
        labelOutputWeights.grid(row=9, column=0, pady=5)

        entryOutputWeights = tkinter.Listbox(root)
        entryOutputWeights.grid(row=10, column=0)

        labelOutputBias = tkinter.Label(text='Output Bias:')
        labelOutputBias.grid(row=9, column=1)

        entryOutputBias = tkinter.Listbox(root)
        entryOutputBias.grid(row=10, column=1, pady=10)

        labelHiddentWeights = tkinter.Label(text='hidden weights:')
        labelHiddentWeights.grid(row=9, column=2, pady=5)

        entryHiddenWeights = tkinter.Listbox(root)
        entryHiddenWeights.grid(row=10, column=2)

        labelHiddenBias = tkinter.Label(text='hidden Bias:')
        labelHiddenBias.grid(row=9, column=3)

        entryHiddenBias = tkinter.Listbox(root)
        entryHiddenBias.grid(row=10, column=3, pady=10)

        figure1 = plt.Figure(figsize=(7,5), dpi=90)
        plot1 = figure1.add_subplot(111)

        plot1.set_ylabel("Loss")
        plot1.set_xlabel("Epochs")
        bar1 = FigureCanvasTkAgg(figure1, master=root)
        bar1.draw()
        bar1.get_tk_widget().grid(row=0, column=2,rowspan=9,columnspan=9)
        


    def sigmoid (self,x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    #Training algorithm
    def train(self):
        self.epochs = self.epoch.get()
        self.lr = self.learningRate.get()
        self.hiddenLayerNeurons=self.hiddenNeurons.get()
        self.inputLayerNeurons = self.entryXVar.get()
        if self.inputLayerNeurons==4:
            self.inputs = np.array([[0,0,0,0],
                                    [0,0,0,1],
                                    [0,0,1,0],
                                    [0,0,1,1],
                                    [0,1,0,0],
                                    [0,1,0,1],
                                    [0,1,1,0],
                                    [0,1,1,1],
                                    [1,0,0,0],
                                    [1,0,0,1],
                                    [1,0,1,0],
                                    [1,0,1,1],
                                    [1,1,0,0],
                                    [1,1,0,1],
                                    [1,1,1,0],
                                    [1,1,1,1]])
            self.expected_output = np.array([[0],
                                            [1],
                                            [1],
                                            [0],
                                            [1],
                                            [0],
                                            [0],
                                            [1],
                                            [1],
                                            [0],
                                            [0],
                                            [1],
                                            [0],
                                            [1],
                                            [1],
                                            [0]])
        elif self.inputLayerNeurons==2:
            self.inputs = np.array([[0,0],
                                    [0,1],
                                    [1,0],
                                    [1,1]])
            self.expected_output = np.array([[0],
                                            [1],
                                            [1],
                                            [0]])
        self.outputLayerNeurons = self.entryYVar.get()
        self.Error=np.zeros((self.epochs,1))

        hidden_weights = np.random.uniform(size=(self.inputLayerNeurons,self.hiddenLayerNeurons))
        hidden_bias =np.random.uniform(size=(1,self.hiddenLayerNeurons))
        output_weights = np.random.uniform(size=(self.hiddenLayerNeurons,self.outputLayerNeurons))
        output_bias = np.random.uniform(size=(1,self.outputLayerNeurons))
        print("Initial hidden weights: ",end='')
        print(*hidden_weights)
        print("Initial hidden biases: ",end='')
        print(*hidden_bias)
        print("Initial output weights: ",end='')
        print(*output_weights)
        print("Initial output biases: ",end='')
        print(*output_bias)
        for _ in range(self.epochs):
            #Forward Propagation
            hidden_layer_activation = np.dot(self.inputs,hidden_weights)
            hidden_layer_activation += hidden_bias
            hidden_layer_output = self.sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output,output_weights)
            output_layer_activation += output_bias
            predicted_output = self.sigmoid(output_layer_activation)

            #Backpropagation
            error = self.expected_output - predicted_output
            d_predicted_output = error * self.sigmoid_derivative(predicted_output)
            self.Error[_]=d_predicted_output[0]

            error_hidden_layer = d_predicted_output.dot(output_weights.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            #Updating Weights and Biases
            output_weights += hidden_layer_output.T.dot(d_predicted_output) * self.lr
            output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) *self.lr
            hidden_weights += self.inputs.T.dot(d_hidden_layer) * self.lr
            hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * self.lr


        figure1 = plt.Figure(figsize=(7,5), dpi=90)
        plot1 = figure1.add_subplot(111)
        plot1.plot(self.Error)
        plot1.set_ylabel("Loss")
        plot1.set_xlabel("Epochs")
        bar1 = FigureCanvasTkAgg(figure1, master=root)
        bar1.draw()
        bar1.get_tk_widget().grid(row=0, column=2,rowspan=9,columnspan=9)

        entryExpectedAnswer = tkinter.Listbox(root)
        entryExpectedAnswer.insert(tkinter.END, *self.expected_output)
        entryExpectedAnswer.grid(row=8, column=0)
        
        entryAnswer = tkinter.Listbox(root)
        entryAnswer.insert(tkinter.END, *predicted_output)
        entryAnswer.grid(row=8, column=1, pady=10)

        entryOutputWeights = tkinter.Listbox(root)
        entryOutputWeights.insert(tkinter.END, *output_weights)
        entryOutputWeights.grid(row=10, column=0)

        entryOutputBias = tkinter.Listbox(root)
        entryOutputBias.insert(tkinter.END, *output_bias)
        entryOutputBias.grid(row=10, column=1, pady=10)

        entryHiddenWeights = tkinter.Listbox(root)
        entryHiddenWeights.insert(tkinter.END, *hidden_weights)
        entryHiddenWeights.grid(row=10, column=2)

        entryHiddenBias = tkinter.Listbox(root)
        entryHiddenBias.insert(tkinter.END, *hidden_bias)
        entryHiddenBias.grid(row=10, column=3, pady=10)


GUI(root)
root.title('Yapay Zekaya Giriş Proje')
root.mainloop()