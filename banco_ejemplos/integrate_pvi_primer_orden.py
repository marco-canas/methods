from sympy import * 
import numpy as np  # librería utilizada para la determinación de las matrices solucion aproximada y error
import pandas as pd # librería utilizada para construir la tabla de comparación de soluciones
from matplotlib import pyplot as plt # graficar la comparación en un plano de la solución exacta y la solución aproximada
import matplotlib.pylab as plt
from pytexit import py2tex, for2tex, for2py 

def conver(yp):
    """
    input:
    yp es una cadena de caracteres que define a y' = F(x,y)
    output:
    yp_sympy es decir, la conversión de la cadena como expresión de sympy
    """
    x,y = symbols('x,y')
    y = Function('y')(x)
    yp_sympy = sympify(yp).subs({'y':y})    # línea importante para establecer a y como función de x en la expresión convertida
    return yp_sympy 

def sol_exact(yp_sympy,X0,Y0):
    """
    Esta función resuelve el PVI y' = F(x,y), y(x0) = y0. 
    Input:
    yp_sympy es y' como expresión de sympy. 
    Output: 
    La función de salida y es una función lambda de Python
    """
    x,y = symbols('x,y')
    y = Function('y')(x)
    eq = Eq(diff(y), yp_sympy)
    solgen = dsolve(eq,y)
    solgen = solgen.subs({O(x**6):0})
    eqc = solgen.subs({x:X0,y:Y0})
    C1 = symbols('C1')
    if len(solve(eqc,C1)) == 1:
        Const = solve(eqc,C1)[0].evalf() 
    else:    # recuerde que la función solve arroja una lista o un diccionario. 
        Const = solve(eqc,C1)[1].evalf() 
    solpvi = solgen.subs({C1:Const}) 
    y = lambdify(x,solpvi.rhs)
    return y,solpvi

def integrate(yp, X0, Y0, paso, a):
    x,y = symbols('x,y') # para cálculo simbólico
        x_,y_ = symbols('x_,y_')  # para cálculo numérico
        y = Function('y')
        yp_sympy = conver(yp)
        ypp_sympy = diff(yp_sympy,x)
        ypp_sympy = ypp_sympy.subs({diff(y,x):yp_sympy}) 
        y_sympy = sol_exact(yp_sympy,X0,Y0)[0]
        yp_numpy = lambdify([x_,y_],yp_sympy.subs({x:x_,y(x):y_}))     # función 1 para cálculo numérico 
                                               #(Ojo: es necesario especificar que nos referimos a y(x))
        d = 0.001    # denominador del cociente incremental 
        ypp_numpy=lambda x_,y_: ((yp_numpy(x_+d,y_)-yp_numpy(x_,y_))/d+((yp_numpy(x_,y_+d)-yp_numpy(x_,y_))/d)*yp_numpy(x_,y_)) 
    
        # función 2 para cálculo numérico
        def y_exac(xn):
            return sol_exact(yp_sympy,X0,Y0)[0](xn)
    
        def y_approx(x0,y0):
            b = yp_numpy(x0, y0)*paso + 0.5*ypp_numpy(x0, y0)*paso**2
            return b
        X = []                                   # Lista de python para los puntos del dominio 
        Y_approx = []                            # Lista de Python que definen la solución aproximada 
        Y_exact = []                             # Lista de Python que define a la solución exacta 
        X.append(X0)
        Y_approx.append(Y0)
        Y_exact.append(Y0)                       # y' = 2xy, donde y = exp(x^2 - 1)
        x0,y0 = X0,Y0    
        while x0 < a:
            paso = min(paso, a - x0)
            y0 += y_approx(x0,y0)
            Y_approx.append(y0)
            x0 += paso 
            y0_exac = round( float( y_exac(x0) ), 4)  
            Y_exact.append(y0_exac)
            X.append(x0)
        
    Error = np.abs(np.array(Y_approx) - np.array(Y_exact))/np.abs(np.array(Y_exact))*100    
    
    table = pd.DataFrame({'X':X, 'Solución aproximada Y':Y_approx, 'Solución exacta Y':Y_exact, 'error en porcentaje':Error})
    
    fig = plt.figure(figsize = (12,8)) # crea la figura

    ax = fig.add_subplot()             # crea los ejes 

    ax.plot(X,Y_approx, label = 'solución aproximada', color = 'blue' )

    ax.plot(X,Y_exact, label = 'solución exacta', color = 'red')
    ax.set(title = r"Comparación entre la solución exacta y la solución aproximada de y' = {} \\ La solución analítica obtenida fue y = {}".format(yp,sol_exact(yp_sympy,X0,Y0)[1] ), 
       xlabel = 'x', ylabel = r'$y(x)$')

    ax.legend() 

    ax.grid()

    # plt.savefig('comparacion_de_soluciones.jpg')  # esta función crea el archivo .jpg en la carpeta de trabajo. 
    return table 

yp = input('Entre F(x,y) como cadena en notación de expresiones de Python: ')
X0 = input('entre x0')
Y0 = input('entre y0')
paso = input('entre el paso h')
a = input("entre el punto 'a' del dominio de y(x) donde quere estimar el valor de y")
integrate(yp, X0, Y0 , paso, a) # a > X0

