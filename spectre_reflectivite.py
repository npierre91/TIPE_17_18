import numpy as np
import matplotlib.pyplot as plt


def couche(lambda0,n,e,theta):
    #Cette fonction calcule la matrice couche. Elle reçoit comme argument la longueur d'onde 
    #dans le vide à laquelle on travaille en nanomètre (lambda), l'indice complexe du milieu (n), 
    #l'épaisseur de la couche en nanomètre (e) et l'angle que fait la direction de propagation 
    #avec la normale des différentes interfaces en radian (theta).
    #Elle retourne une matrice 2*2 correspondant à la matrice couche.

    #Calcul du vecteur d'onde dans le vide
    k0=2*np.pi/lambda0
    #Calcul du vecteur d'onde dans le milieu
    k=k0*n
    #Calcul de la composante selon (Oz)du vecteur d'onde dans le milieu
    kz=k*np.cos(theta)
    #Calcul du retard de phase
    beta=kz*e
    
    #Calcul de la matrice couche
    C=np.zeros([2,2], dtype=np.complex)
    C[0,0],C[1,1]=np.exp(1j*beta),np.exp(-1j*beta)
    return C


def interfaceTE(n1,theta1,n2,theta2):   
#Cette fonction calcule la matrice interface en polarisation TE correspondant à l'interface 
#entre un milieu supérieur d'indice complexe n1 et un milieu inférieur d'indice complexe n2.
#Elle reçoit comme argument n1, n2, theta1 (angle que fait la direction de propagation de 
#l'onde dans le milieu 1 par rapport à la normale à l'interface) et theta2 (angle que fait 
#la direction de propagation de l'onde dans le milieu 2 par rapport à la normale à l'interface)
#Elle retoune une matrice 2*2 correspondant à la matrice interface.

    #Calcul du coefficient de réflexion en polarisation TE
    r_TE=(n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2))
    #Calcul du coefficient de transmission en polarisation TE
    t_TE=2*n1*np.cos(theta1)/(n1*np.cos(theta1)+n2*np.cos(theta2))

    #Calcul de la matrice interface
    I=np.ones([2,2], dtype=np.complex)
    I=I/t_TE
    I[0,1],I[1,0]=I[0,1]*r_TE,I[1,0]*r_TE
    return I

def spectres_reflectivite(theta,e_Al,e_PMMA,e_silice):
    #Epaisseurs des couches du miroir diélectrique :
    e_SiO=59.5
    e_SiN=45.1
    #On définit la plage d'énergie en eV sur laquelle on veut travailler
    E=np.arange(2.8,4.4,0.001)
    #On la transforme en terme de longueur d'onde dans le vide en nm
    lambda_0=1240./E
    #On note N le nombre de points de mesure
    N=len(lambda_0)
    #On définit les différents indices mis en jeu
    n_air = 1
    n_Al = 0.35 - 4.1j
    n_PMMA = 1.49
    n_silice=1.485
    #Pour le miroir diélectrique :
    n_SiN = 1.96
    n_SiO = 1.485
    n_substrat = 1.47865
    #On définit les angles dans chaque milieu que fait la direction de propagation 
    #avec la normale
    theta_air = theta
    theta_substrat = np.arcsin(n_air*np.sin(theta_air)/n_substrat)
    #Pour le miroir diélectrique :
    theta_SiN = np.arcsin(n_air*np.sin(theta_air)/n_SiN)
    theta_SiO = np.arcsin(n_air*np.sin(theta_air)/n_SiO)

    theta_silice = np.arcsin(n_air*np.sin(theta_air)/n_silice)
    theta_PMMA = np.arcsin(n_air*np.sin(theta_air)/n_PMMA)
    theta_Al = np.arcsin(n_air*np.sin(theta_air)/n_Al)
    #On calcule les matrices interface mises en jeu pour la polarisation TE
    I_substrat_vers_SiN_TE = interfaceTE(n_substrat,theta_substrat,n_SiN,theta_SiN)
    I_SiN_vers_SiO_TE = interfaceTE(n_SiN,theta_SiN,n_SiO,theta_SiO)
    I_SiO_vers_SiN_TE = interfaceTE(n_SiO,theta_SiO,n_SiN,theta_SiN)
    I_SiN_vers_silice_TE = interfaceTE(n_SiN,theta_SiN,n_silice,theta_silice)
    I_silice_vers_PMMA_TE = interfaceTE(n_silice,theta_silice,n_PMMA,theta_PMMA)
    I_PMMA_vers_Al_TE = interfaceTE(n_PMMA,theta_PMMA,n_Al,theta_Al)
    I_Al_vers_air_TE = interfaceTE(n_Al,theta_Al,n_air,theta_air)
    
    #On prépare les vecteurs réflexion et transmission
    R_TE=np.zeros(N)
    T_TE=np.zeros(N)

    for i in range(1,N):
        #On calcule les différentes matrices couches
        C_SiN = couche(lambda_0[i], n_SiN, e_SiN, theta_SiN)
        C_SiO = couche(lambda_0[i], n_SiO, e_SiO, theta_SiO)
        C_silice = couche(lambda_0[i], n_silice, e_silice, theta_silice)
        C_PMMA = couche(lambda_0[i], n_PMMA, e_PMMA, theta_PMMA)
        C_Al = couche(lambda_0[i], n_Al, e_Al, theta_Al)

        S_TE=np.dot(I_substrat_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_SiO_TE,np.dot(C_SiO,np.dot(I_SiO_vers_SiN_TE,np.dot(C_SiN,np.dot(I_SiN_vers_silice_TE,np.dot(C_silice,np.dot(I_silice_vers_PMMA_TE,np.dot(C_PMMA,np.dot(I_PMMA_vers_Al_TE,np.dot(C_Al,I_Al_vers_air_TE))))))))))))))))))))))))))))))))))))))))
        R_TE[i]=abs(S_TE[1,0]/S_TE[0,0])**2
        T_TE[i]=n_air/n_substrat*abs(1/S_TE[0,0])**2
    plt.plot(E,R_TE,'k')
    plt.show()
