import os
import psutil
import time
import subprocess
import fnmatch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.ttk import *
from PIL import ImageFilter,Image
from tkinter import filedialog, messagebox
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# loading Python Imaging Library
from PIL import ImageTk, Image
main_win = Tk()

data=pd.read_csv('food.csv')
Breakfastdata=data['Breakfast']
BreakfastdataNumpy=Breakfastdata.to_numpy()
    
Lunchdata=data['Lunch']
LunchdataNumpy=Lunchdata.to_numpy()

global bmivariable
Dinnerdata=data['Dinner']
DinnerdataNumpy=Dinnerdata.to_numpy()
Food_itemsdata=data['Food_items']

def get_bmi():
    global bmivariable
    """
    Takes an image performs image processing using PIL to get BMI
    """
    # Select the Imagename from a folder 
    img_content = openfilename()

    img = Image.open(img_content).convert('LA')
    img1 = Image.open(img_content)
    thresh = 250
    fn = lambda x : 0 if x > thresh else 255
    r = img.convert('L').point(fn, mode='1')
    current_width, current_height = r.size
    BMI_list = []

    for i in range(0,8):
        new_width = round(current_width - (current_width * 0.1))
        new_height = round(current_height - (current_height * 0.1))
        r = r.resize((new_width, new_height), Image.LANCZOS)
        r.save('resize_result/'+str(i)+'.png')
        current_width = new_width
        current_height = new_height
        # print(current_width,current_height)
        pix_val = list(r.getdata())

        #Calculating the Area 
        area = 0 
        for val in pix_val:
            if val == 255:
                area = area + 1

        # print(f"The area of the silhoutte is {area}")

        #Calculating the Height of the person
        iar = np.asarray(r)
        obj_h = 0 
        for x in iar:
            for y in x:
                if y:
                    obj_h = obj_h + 1
                    break
        H = obj_h
        #Calculating the BMI
        pie = 22/7
        pie = round(pie,4)

        BMI_img = ((pie * (area**2))/(8*(H**3)) - 4.1219)/0.1963
        #CBMI = (BMI_img - 4.123)/0.1963
        print("\n Pie: %s\n Area: %s Height: %s, BMI_img: %s ",pie, area, H, BMI_img)

        BMI_list.append(BMI_img)
	# resize the image and apply a high-quality down sampling filter
    img = img1.resize((250, 250), Image.LANCZOS)
	# PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img1)
    
    bmivariable = round(np.mean(BMI_list),2)
    print("\n ibmi mean: %s",bmivariable)
    #e7.setvar(bmivariable)
    panel = Label(main_win, image = img)
	# set the image as img 
    panel.image = img
    panel.grid(row = 5, column=0,padx=0, pady=4 )
    panel = Label(main_win, text= bmivariable)
	# set the image as img 
    panel.image = img
    panel.grid(row = 5, column=0,padx=0, pady=4)
    panel = Label(main_win, text= bmivariable)
    #bmivariable = round(np.mean(BMI_list),2)
    return bmivariable

def openfilename(): 
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='"Chose Jpg Image')
    return filename

def open_img():
	# Select the Imagename from a folder 
	x = openfilename()
	# opens the image
	img1 = Image.open(x)
	# resize the image and apply a high-quality down sampling filter
	img = img1.resize((250, 250), Image.LANCZOS)
	# PhotoImage class is used to add image to widgets, icons etc
	img = ImageTk.PhotoImage(img)
	# create a label
	panel = Label(main_win, image = img)
	# set the image as img 
	panel.image = img
	panel.grid(row = 5, column=0,padx=1, pady=4)
   

def show_entry_fields():
    print("\n Age: %s\n Weight: %s kg\n Hight: %s cm\n" % (e1.get(), e2.get(),e3.get()))

def Weight_Loss_Image():
    global bmivariable
    show_entry_fields()
    e5.delete('1.0', END)
   
    e5.insert('end', "\n Age: %s\n Weight: %s kg\n Hight: %s cm\n" % (e1.get(), e2.get(),e3.get()), True)
  
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
        
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
        
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)
        
    # retrieving Lunch data rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    #print(LunchfoodseparatedIDdata)
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    #print(LunchfoodseparatedIDdata)

    # retrieving Breafast data rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
        
        
    # retrieving Dinner Data rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
        
    #calculating BMI
    age=int(e1.get())
    #veg=float(e2.get())
    weight=float(e2.get())
    height=float(e3.get())
    ibmi = bmivariable
    #bmi = weight/((height/100)**2) 
    bmi = round(weight/((height/100)**2),2)
    agewiseinp=0
        
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    

        
    #conditions
    print("Your body mass index is: ", ibmi)
    print("Your body mass index is: ", bmi)

    e5.insert('end', "\n Your Image body mass index is: %s\n" % ibmi, True)
    e5.insert('end', "\n Your body mass index is: %s\n" % bmi, True)
    if ( bmi < 16):
        print("Acoording to your BMI, you are Severely Underweight")
        e5.insert('end', "\n Acoording to your BMI, you are Severely Underweight\n", True)
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("Acoording to your BMI, you are Underweight")
        e5.insert('end', "\n Acoording to your BMI, you are Underweight\n", True)
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Acoording to your BMI, you are Healthy")
        e5.insert('end', "\n Acoording to your BMI, you are Healthy\n", True)
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("Acoording to your BMI, you are Overweight")
        e5.insert('end', "\n Acoording to your BMI, you are Overweight\n", True)
        clbmi=1
    elif ( bmi >=30):
        print("Acoording to your BMI, you are Severely Overweight")
        e5.insert('end', "\n Acoording to your BMI, you are Severely Overweight\n", True)
        clbmi=0

    #converting into numpy array
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(clbmi+agecl)/2
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    XValu=np.arange(0,len(kmeans.labels_))
    
    # retrieving the labels for dinner food
    dnrlbl=kmeans.labels_

    ## K-Means Based  lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    
    # retrieving the labels for lunch food
    lnchlbl=kmeans.labels_
    
    ## K-Means Based  lunch Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    
    # retrieving the labels for breakfast food
    brklbl=kmeans.labels_
    
    inp=[]
    ## Reading of the Dataet
    datafin=pd.read_csv('nutrition_distriution.csv')

    ## train set
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    
    X_test=np.zeros((len(weightlosscat),6),dtype=np.float32)

    print('####################')
    
    #randomforest
    for jj in range(len(weightlosscat)):
        valloc=list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    
    
    X_train=weightlossfin# Features
    y_train=yt # Labels

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    #print (X_test[1])
    X_test2=X_test
    y_pred=clf.predict(X_test)
    
    
    print ('SUGGESTED FOOD ITEMS ::')
    e5.insert('end', "\n SUGGESTED FOOD ITEMS FOR WEIGHT LOSS::\n", True)
    for ii in range(len(y_pred)):
        print (Food_itemsdata[ii])
        print (y_pred[ii])
        if y_pred[ii]==2:     #weightloss
            print (Food_itemsdata[ii])
            e5.insert('end', "%s \n" % Food_itemsdata[ii] , True)
            findata=Food_itemsdata[ii]
            

    print('\n Thank You for taking our recommendations. :)')
    e5.insert('end', "\n Thank You for taking our recommendations. ::\n", True)


def Weight_Gain():
    global bmivariable
    show_entry_fields()
    e5.delete('1.0', END)
   
    e5.insert('end', "\n Age: %s\n Weight: %s kg\n Hight: %s cm\n" % (e1.get(), e2.get(),e3.get()), True)
  
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
        
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
        
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)
        
    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
        
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
        
        
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
        
    #claculating BMI
    age=int(e1.get())
    #veg=float(e2.get())
    weight=float(e2.get())
    height=float(e3.get())
    bmi = round(weight/((height/100)**2),2)
    ibmi = bmivariable    

    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)

    print("Your image body mass index is: ", ibmi)
    print("Your body mass index is: ", bmi)
    e5.insert('end', "\n Your image body mass index is: %s \n" % (ibmi), True)
    e5.insert('end', "\n Your body mass index is: %s \n" % (bmi), True)
    if ( bmi < 16):
        print("Acoording to your BMI, you are Severely Underweight")
        e5.insert('end', "\n Acoording to your BMI, you are Severely Underweight \n" , True)
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("Acoording to your BMI, you are Underweight")
        e5.insert('end', "\n Acoording to your BMI, you are Underweight \n" , True)
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Acoording to your BMI, you are Healthy")
        e5.insert('end', "\n Acoording to your BMI, you are Healthy \n" , True)
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("Acoording to your BMI, you are Overweight")
        e5.insert('end', "\n Acoording to your BMI, you are Overweight \n" , True)
        clbmi=1
    elif ( bmi >=30):
        print("Acoording to your BMI, you are Severely Overweight")
        e5.insert('end', "\n Acoording to your BMI, you are Severely Overweight \n" , True)
        clbmi=0


    
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # plt.bar(XValu,kmeans.labels_)
    dnrlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    
    ## K-Means Based  lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    lnchlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    
    ## K-Means Based  lunch Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    brklbl=kmeans.labels_
    
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    inp=[]
    ## Reading of the Dataet
    datafin=pd.read_csv('nutrition_distriution.csv')
    datafin.head(5)
    
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
   
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            #print (valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    
    X_test=np.zeros((len(weightgaincat),10),dtype=np.float32)

    print('####################')
    # In[287]:
    for jj in range(len(weightgaincat)):
        valloc=list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    
    X_train=weightgainfin# Features
    y_train=yr # Labels
    
   
    
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
   
    X_test2=X_test
    y_pred=clf.predict(X_test)
    
    print (len(X_test))
    
    
    print ('SUGGESTED FOOD ITEMS ::')
    e5.insert('end', "\n SUGGESTED FOOD ITEMS FOR WEIGHT GAIN:: \n" , True)
    print (len(y_pred))
    for ii in range(len(y_pred)):
        print (Food_itemsdata[ii])
        print (y_pred[ii])
        if y_pred[ii]==2:
            print (Food_itemsdata[ii])
            e5.insert('end', "%s \n" % Food_itemsdata[ii], True)
            findata=Food_itemsdata[ii]
            if int(veg)==1:
                datanv=['Chicken Burger']
                for it in range(len(datanv)):
                    if findata==datanv[it]:
                        print('VegNovVeg')

    print('\n Thank You for taking our recommendations. :)')
    e5.insert('end', "\n Thank You for taking our recommendations. ::\n", True)



def Healthy():
    global bmivariable
    show_entry_fields()
    e5.delete('1.0', END)
   
    e5.insert('end', "\n Age: %s\n Weight: %s kg\n Hight: %s cm\n" % (e1.get(), e2.get(),e3.get()), True)
    
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
        
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
        
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)
        
    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
        
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
        
        
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
        
    
    age=int(e1.get())
    #veg=float(e2.get())
    weight=float(e2.get())
    height=float(e3.get())
    #bmi = weight/((height/100)**2) 
    ibmi =  bmivariable
    bmi = round(weight/((height/100)**2),2)
    agewiseinp=0
        
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    

        
    #conditions
    print("Your Image body mass index is: ", ibmi)
    print("Your body mass index is: ", bmi)
    e5.insert('end', "\n Your Image body mass index is: %s \n" % (ibmi), True)
    e5.insert('end', "\n Your body mass index is: %s \n" % (bmi), True)
    
    if ( bmi < 16):
        print("Acoording to your BMI, you are Severely Underweight")
        e5.insert('end', "\n Acoording to your BMI, you are Severely Underweight", True)
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        print("Acoording to your BMI, you are Underweight")
        e5.insert('end', "\n Acoording to your BMI, you are Underweight", True)
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        print("Acoording to your BMI, you are Healthy")
        e5.insert('end', "\n Acoording to your BMI, you are Healthy", True)
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        print("Acoording to your BMI, you are Overweight")
        e5.insert('end', "\n Acoording to your BMI, you are Overweight", True)
        clbmi=1
    elif ( bmi >=30):
        print("Acoording to your BMI, you are Severely Overweight")
        e5.insert('end', "\n Acoording to your BMI, you are Severely Overweight", True)
        clbmi=0

    
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    

    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    dnrlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    #print ('## Prediction Result ##')
    #print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    lnchlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
   
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    brklbl=kmeans.labels_
    # print (len(brklbl))
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    inp=[]
    ## Reading of the Dataet
    datafin=pd.read_csv('nutrition_distriution.csv')
    datafin.head(5)
   
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            #print (valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    
    for jj in range(len(healthycat)):
        valloc=list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    
    X_train=healthycatfin# Features
    y_train=ys # Labels
    
    
    
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    
    X_test2=X_test
    y_pred=clf.predict(X_test)
   
    
    
    print ('SUGGESTED FOOD ITEMS ::')
    e5.insert('end', "\n SUGGESTED FOOD ITEMS FOR HEALTHY::\n", True)
    for ii in range(len(y_pred)):
        print (Food_itemsdata[ii])
        print (y_pred[ii])
        if y_pred[ii]==2 and ii < 89:
            
            e5.insert('end', " %s\n" %(Food_itemsdata[ii]), True)
            findata=Food_itemsdata[ii]
            if int(veg)==1:
                datanv=['Chicken Burger']

    print('\n Thank You for taking our recommendations. :)')
    e5.insert('end', "\n Thank You for taking our recommendations. ::\n", True)



if __name__ == '__main__':

    Label(main_win,text="Age").grid(row=0,column=0,sticky=W,pady=4, padx=10)
    #Label(main_win,text="veg/Non veg (1/0)").grid(row=1,column=0,sticky=W,pady=4, padx=10)
    Label(main_win,text="Weight (in kg)").grid(row=1,column=0,sticky=W,pady=4, padx=10)
    Label(main_win,text="Height (in cm)").grid(row=2,column=0,sticky=W,pady=4, padx=10)
    Label(main_win, text='Upload Image to calculate BMI in jpg format ').grid(row=3, column=0,sticky=W,pady=4, padx=10)
    e1 = Entry(main_win)
    e2 = Entry(main_win)
    e3 = Entry(main_win)
    #e4 = Entry(main_win)
    e5 = Text(main_win, height=30,width=40)

    e1.grid(row=0, column=0)
    e2.grid(row=1, column=0)
    e3.grid(row=2, column=0)
    #e4.grid(row=3, column=0)
    e5.grid(row=5, column=1,padx=0, sticky=W,pady=0)

    Button(main_win,text='Quit',command=main_win.quit).grid(row=0,column=3,sticky=W,padx=0,pady=4)
    Button(main_win,text='Weight Loss',command=Weight_Loss_Image).grid(row=0,column=1,sticky=W,padx=10,pady=4)
    Button(main_win,text='Weight Gain',command=Weight_Gain).grid(row=1,column=1,sticky=W,padx=10,pady=4)
    Button(main_win,text='Healthy',command=Healthy).grid(row=2,column=1,sticky=W,padx=10,pady=4)
    Button(main_win,text='Chose File & Calc BMI',command=get_bmi).grid(row=3,column=1,sticky=W,padx=10, pady=4)

    main_win.geometry("1000x1000")
    main_win.wm_title("DIET RECOMMENDATION SYSTEM")

    main_win.mainloop()
