from os import write
import re
from flask import Flask, render_template, request, send_file
from matplotlib.pyplot import plot
from pandas.core.frame import DataFrame

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/mediamm')
def mediamm():
    return render_template('mediamm.html')


@app.route('/cuadradosMedios')
def cuadradosMedios():
    return render_template('cuadradosMedios.html')


@app.route('/printCuadradosMedio')
def imprimirCuadradosMedios():
    return render_template('printCuadradosMedios.html')


@app.route('/calcularCuadradosMedios', methods=['GET', 'POST'])
def calcularCuadradosMedios():
    n = request.form.get('numeroIteraciones', type=int)
    r = request.form.get('semilla', type=int)

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # n=100
    # r=23456
    l = len(str(r))
    lista = []
    lista2 = []
    i = 1
    while i <= n:
        x = str(r*r)
        if l % 2 == 0:
            x = x.zfill(l*2)
        else:
            x = x.zfill(l)
        y = (len(x)-l)/2
        y = int(y)
        r = int(x[y:y+l])
        lista.append(r)
        lista2.append(x)
        i = i+1
    df = pd.DataFrame({'X2': lista2, 'Xi': lista})
    dfrac = df["Xi"]/10**l
    df['ri'] = dfrac

    buf = io.BytesIO()
    x1 = df['ri']
    plt.plot(x1)
    plt.title('Generador de Números Aleatorios Cuadrados Medios')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()
            
    df.to_csv("static/file/data.csv", index=False) """

    return render_template('printCuadradosMedios.html', data=data, image=plot_url)


@app.route('/congruencialLineal')
def congruencialLineal():
    return render_template('congruencialLineal.html')


@app.route('/printCongruencialLineal')
def imprimirCongruencialLineal():
    return render_template('printCongruencialLineal.html')


@app.route('/calcularCongruencialLineal', methods=['GET', 'POST'])
def calcularCongruencialLineal():
    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    #n, m, a, x0, c = 20,1000,101,4,457
    x = [1]*n
    r = [0.1]*n
    for i in range(0, n):
        x[i] = ((a*x0)+c) % m
        x0 = x[i]
        r[i] = x0/m
    df = pd.DataFrame({'Xn': x, 'ri': r})

    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(r, marker='o')
    plt.title('Generador de Números Aleatorios Congruencial Lineal')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()
            
    df.to_csv("static/file/data.csv", index=False)  """

    return render_template('printCongruencialLineal.html', data=data, image=plot_url)


@app.route('/congruencialMultiplicativo')
def congruencialMultiplicativo():
    return render_template('congruencialMultiplicativo.html')


@app.route('/printCongruencialMultiplicativo')
def imprimirCongruencialMultiplicativo():
    return render_template('printCongruencialMultiplicativo.html')


@app.route('/calcularCongruencialMultiplicativo', methods=['GET', 'POST'])
def calcularCongruencialMultiplicativo():
    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * n
    r = [0.1] * n
    for i in range(0, n):
        x[i] = (a*x0) % m
        x0 = x[i]
        r[i] = x0 / m
    d = {'Xn': x, 'ri': r}
    df = pd.DataFrame(data=d)

    buf = io.BytesIO()
    plt.plot(r, 'g-', marker='o',)
    plt.title('Generador de Números Aleatorios Congruencial Multiplicativo')
    plt.xlabel('Serie')
    plt.ylabel('Aleatorios')
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()          
    
    df.to_csv("static/file/data.csv", index=False) """

    return render_template('printCongruencialMultiplicativo.html', data=data, image=plot_url)


@app.route('/calcularMediaModaMediana', methods=['GET', 'POST'])
def calcularMediaModaMediana():
    columna = request.form.get("nombreColumna")

    file = request.files['file'].read()

    # importamos la libreria Pandas, matplotlib y numpy que van a ser de mucha utilidad para poder hacer gráficos
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    from pandas import DataFrame

    # leemos los datos de la tabla del directorio Data de trabajo
    datos = pd.read_excel(file)
    # Presentamos los datos en un DataFrame de Pandas
    datos

    # Preparando para el grafico para la columna TOTAL PACIENTES
    buf = io.BytesIO()
    x = datos[columna]
    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=8, color='blue')
    plt.axvline(x.mean(), color='red', label='Media')
    plt.axvline(x.median(), color='yellow', label='Mediana')
    plt.axvline(x.mode()[0], color='green', label='Moda')
    plt.xlabel('Total de datos')
    plt.ylabel('Frecuencia')
    plt.legend()

    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    media = datos[columna].mean()
    moda = datos[columna].mode()
    mediana = datos[columna].median()

    df = pd.DataFrame(columns=('Media', 'Moda', 'Mediana'))
    df.loc[len(df)] = [media, moda, mediana]
    df
    data = df.to_html(classes="table table-striped",
                      justify="justify-all", border=0)

    # Tomamos los datos de las columnas
    df2 = datos[[columna]].describe()
    # describe(), nos presenta directamente la media, desviación standar, el valor mínimo, valor máximo, el 1er cuartil, 2do Cuartil, 3er Cuartil
    data2 = df2.to_html(classes="table table-ligth table-striped",
                        justify="justify-all", border=0)

    return render_template('printMediaMedianaModa.html', data=data, data2=data2, image=plot_url)


@app.route('/promedioMovil')
def promedioMovil():
    return render_template('promedioMovil.html')


@app.route('/printPromedioMovil')
def printPromedioMovil():
    return render_template('printPromedioMovil.html')


@app.route('/calcularPromedioMovil', methods=['GET', 'POST'])
def calcularPromedioMovil():
    file = request.files['file'].read()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # el DataFrame se llama movil
    # exporta = {'Año':[2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],
    #           'Exportaciones':[5501.0, 6232.7, 8118.3, 10137.00, 10449.50, 12794.60, 9939.10, 13193.00, 16036.2, 18496.90, 18709.30, 19363.50, 16521.50, 15175.40, 16927.00]}

    file = pd.read_excel(file)
    name = file.columns
    columna1 = name[0]
    columna2 = name[1]
    movil = pd.DataFrame(file)
    movil.head()


# calculamos para la primera media móvil MMO_3
    for i in range(0, movil.shape[0]-2):
        movil.loc[movil.index[i+2], 'MMO_3'] = np.round(
            ((movil.iloc[i, 1]+movil.iloc[i+1, 1]+movil.iloc[i+2, 1])/3), 1)
    # calculamos para la segunda media móvil MMO_4
    for i in range(0, movil.shape[0]-3):
        movil.loc[movil.index[i+3], 'MMO_4'] = np.round(((movil.iloc[i, 1]+movil.iloc[i+1, 1]+movil.iloc[i+2, 1]+movil.iloc[i +
                                                                                                                            3, 1])/4), 1)
    # calculamos la proyeción final
    proyeccion = movil.iloc[12:, [1, 2, 3]]
    p1, p2, p3 = proyeccion.mean()
    # print(p1,p2,p3)
    # incorporamos al DataFrame
    df = movil.append({columna1: columna1, columna2: p1,
                      'MMO_3': p2, 'MMO_4': p3}, ignore_index=True)
    # mostramos los resultados
    df['e_MM3'] = df[columna2]-df['MMO_3']
    df['e_MM4'] = df[columna2]-df['MMO_4']
    df

    buf = io.BytesIO()
    # plt.figure(figsize=[8,8])
    plt.grid(True)
    plt.plot(df[columna2], label=columna2, marker='o')
    plt.plot(df['MMO_3'], label='Media Móvil 3' + columna1)
    plt.plot(df['MMO_4'], label='Media Móvil 4' + columna1)
    plt.legend(loc=2)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter(data)
    df.to_excel(writer, index=False)
    writer.save()          
    
    df.to_csv(data, index=False)  """

    return render_template('printPromedioMovil.html', data=data, image=plot_url)

    # SUAVIZACIÓN EXPONENCIAL


@app.route('/suavizacionExponencial')
def suavizacionExponencial():
    return render_template('suavizacionExponencial.html')


@app.route('/printSuavizacionExponencial')
def imprimirSuavizacionExponencial():
    return render_template('imprimirSuavizacionExponencial.html')


@app.route('/calcularSuavizacionExponencial', methods=['GET', 'POST'])
def calcularSuavizacionExponencial():
    file = request.files['file'].read()

    # Librerías
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # *** Leer el archivo ***
    file = pd.read_excel(file)
    movil = pd.DataFrame(file)
    name = file.columns
    columna1 = name[0]
    columna2 = name[1]

    movil.head()
    alfa = 0.1
    unoalfa = 1. - alfa
    for i in range(0, movil.shape[0]-1):
        movil.loc[movil.index[i+1], 'SN'] = np.round(movil.iloc[i, 1], 1)
    for i in range(2, movil.shape[0]):
        movil.loc[movil.index[i], 'SN'] = np.round(
            movil.iloc[i-1, 1], 1)*alfa + np.round(movil.iloc[i-1, 2], 1)*unoalfa
    i = i+1
    p1 = 0
    p2 = np.round(movil.iloc[i-1, 1], 1)*alfa + \
        np.round(movil.iloc[i-1, 2], 1)*unoalfa
    df = movil.append({columna1: columna1, columna2: p1,
                      'SN': p2}, ignore_index=True)
    df

    buf = io.BytesIO()  # La clase base de la jerarquía de clases.
    # plt.figure(figsize=[8,8])
    plt.grid(True)
    plt.plot(df[columna2], label=columna2, marker='o')
    plt.plot(df['SN'], label='SN')
    plt.legend(loc=2)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()

    df.to_csv("static/file/data.csv", index=False) """

    return render_template('printSuavizacionExponencial.html', data=data, image=plot_url)


@app.route('/regresionLineal')
def regresionLineal():
    return render_template('regresionLineal.html')


@app.route('/printRegresionLineal')
def imprimirRegresionLineal():
    return render_template('printRegresionLineal.html')


@app.route('/calcularRegresionLineal', methods=['GET', 'POST'])
def calcularRegresionLineal():
    file = request.files['file'].read()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # el DataFrame se llama movil
    # exporta = {'Año':[2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],
    #           'Exportaciones':[5501.0, 6232.7, 8118.3, 10137.00, 10449.50, 12794.60, 9939.10, 13193.00, 16036.2, 18496.90, 18709.30, 19363.50, 16521.50, 15175.40, 16927.00]}
    file = pd.read_excel(file)
    a = pd.DataFrame(file)
    x = a["X"]
    y = a["Y"]
    
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    p = np.polyfit(x, y, 1)  # 1 para lineal, 2 para polinomio ...
    p0, p1 = p

    xx = x**2
    xx
    # multiplicacion de X e Y
    xy = x*y
    xy
    # Y al cuadrado
    yy = y**2
    yy
    df2 = pd.DataFrame({"X": x, "Y": y, "XX": xx, "XY": xy, "YY": yy})
    df2

    total1 = df2['X'].sum()
    total1
    total2 = df2['Y'].sum()
    total2
    total3 = df2['XX'].sum()
    total3
    total4 = df2['XY'].sum()
    total4
    total5 = df2['YY'].sum()
    total5

    df = pd.DataFrame({"X": x, "Y": y, "XX": xx, "XY": xy, "YY": yy})
    df.loc['Sumatoria'] = [total1, total2, total3, total4, total5]
    df
    # y(x) = poX + p1 = 49.53676471X -242.41911765
    # calculamos los valores ajustados y_ajuste
    buf = io.BytesIO()
    y_ajuste = p[0]*x + p[1]
    print(y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos = plt.plot(x, y, 'b.')
    # Dibujamos la recta de ajuste
    p_ajuste = plt.plot(x, y_ajuste, 'r-')
    plt.title('Ajuste lineal por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales', 'Ajuste lineal',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()

    df.to_csv("static/file/data.csv", index=False) """

    return render_template('printRegresionLineal.html', data=data, image=plot_url)


@app.route('/regresionLinealCuadrada')
def regresionLinealCuadrada():
    return render_template('regresionLinealCuadrada.html')


@app.route('/printRegresionLinealCuadrada')
def imprimirRegresionLinealCuadrada():
    return render_template('iprintRegresionLinealCuadrada.html')


@app.route('/calcularRegresionLinealCuadrada', methods=['GET', 'POST'])
def calcularRegresionLinealCuadrada():
    file = request.files['file'].read()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64

    # el DataFrame se llama movil
    # exporta = {'Año':[2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017],
    #           'Exportaciones':[5501.0, 6232.7, 8118.3, 10137.00, 10449.50, 12794.60, 9939.10, 13193.00, 16036.2, 18496.90, 18709.30, 19363.50, 16521.50, 15175.40, 16927.00]}
    file = pd.read_excel(file)
    a = pd.DataFrame(file)
    x = a["X"]
    y = a["Y"]
    # ajuste de la recta (polinomio de grado 1 f(x) = ax + b)
    pp = np.polyfit(x, y, 2)
    pp0, pp1, pp2 = pp

    xx2 = x**2
    xx2
    xx3 = x**3
    xx3
    xx4 = x**4
    xx4
    # multiplicacion de X e Y
    xy = x*y
    xy
    # Y al cuadrado
    yy = xx2*y
    yy

    df3 = pd.DataFrame({"X": x, "Y": y, "XX": xx2, "X3": xx3,
                       "X4": xx4, "XY": xy, "X2Y": yy})
    df3

    total1 = df3['XX'].sum()
    total1
    total2 = df3['X3'].sum()
    total2
    total3 = df3['X4'].sum()
    total3
    total4 = df3['XY'].sum()
    total4
    total5 = df3['X2Y'].sum()
    total5
    total6 = df3['X'].sum()
    total6
    total7 = df3['Y'].sum()
    total7

    df = pd.DataFrame({"X": x, "Y": y, "XX": xx2, "X3": xx3,
                      "X4": xx4, "XY": xy, "X2Y": yy})
    df.loc['Sumatoria'] = [total6, total7,
                           total1, total2, total3, total4, total5]
    df
    # y(x) = poX + p1 = 49.53676471X -242.41911765
    # calculamos los valores ajustados y_ajuste
    buf = io.BytesIO()
    # calculamos los valores ajustados y_ajuste
    y_ajuste = pp[0]*x*x + pp[1]*x + pp[2]
    print(y_ajuste)
    # dibujamos los datos experimentales de la recta
    p_datos = plt.plot(x, y, 'b.')
    # Dibujamos la curva de ajuste
    p_ajuste = plt.plot(x, y_ajuste, 'r-')
    plt.title('Ajuste Polinomial por mínimos cuadrados')
    plt.xlabel('Eje x')
    plt.ylabel('Eje y')
    plt.legend(('Datos experimentales', 'Ajuste Polinomial',), loc="upper left")
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    df.to_excel(writer, index=False)
    writer.save()

    df.to_csv("static/file/data.csv", index=False) """

    return render_template('printRegresionLinealCuadrada.html', data=data, image=plot_url)


@app.route('/sistemaMontecarlo')
def sistemaMontecarlo():
    return render_template('sistemaMontecarlo.html')


@app.route('/printSistemaMontecarlo')
def printSistemaMontecarlo():
    return render_template('printSistemaMontecarlo.html')


@app.route('/calcularMontecarlo', methods=['GET', 'POST'])
def calcularMontecarlo():
    n = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    file = request.files['file'].read()

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    from pandas import DataFrame

    file = pd.read_excel(file)
    tot = pd.DataFrame(file)
    #x = a["X"]
    #tot = a["Y"]

    # Generamos los 62 # aleatorios por cualquiera de los métodos estudiados y luego podemos realizar la simulación
    # Generador de números aleatorios Congruencia lineal
    # Borland C/C++ xi+1=22695477xi + 1 mod 2^32
    #n, m, a, x0, c = 52, 2**32, 22695477, 4, 1
    x = [1] * n
    r = [0.1] * n
    for i in range(0, n):
     x[i] = ((a*x0)+c) % m
     x0 = x[i]
     r[i] = x0 / m
    # llenamos nuestro DataFrame
    d = {'ri': r}
    dfMCL = pd.DataFrame(data=d)
    dfMCL

    # Ordenamos por Día
    suma = tot['Y'].sum()
    n = len(tot)
    suma
    x1 = tot.assign(Probabilidad=lambda x: x['Y'] / suma)
    x2 = x1.sort_values('X')
    a = x2['Probabilidad']
    a

    a1 = np.cumsum(a)  # Cálculo la suma acumulativa de las probabilidades
    x2['FPA'] = a1
    x2

    x2['Min'] = x2['FPA']
    x2['Max'] = x2['FPA']
    x2

    lis = x2["Min"].values
    lis2 = x2['Max'].values
    lis[0] = 0
    for i in range(1, 7):
     lis[i] = lis2[i-1]
    x2['Min'] = lis
    x2

    max = x2['Max'].values
    min = x2['Min'].values

    x2

    x2 = pd.DataFrame(x2)

    data = x2.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    def busqueda(arrmin, arrmax, valor):
        #print(valor)
        for i in range(len(arrmin)):
            # print(arrmin[i],arrmax[i])
            if valor >= arrmin[i] and valor <= arrmax[i]:
                return i

        return -1
    xpos = dfMCL['ri']
    posi = [0] * n

    for j in range(n):
        val = xpos[j]
        pos = busqueda(min, max, val)
        posi[j] = pos
    x2 = x2.astype({"X": int})

    import itertools
    import math
    simula = []
    for j in range(n):
        for i in range(n):
            sim = x2.loc[x2["X"] == posi[i]+1]
            simu = sim.filter(['Y']).values
            iterator = itertools.chain(*simu)
            for item in iterator:
                a = item
            simula.append(round(a, 2))
    simula

    dfMCL["Simulación"] = pd.DataFrame(simula)
    dfMCL["Costo de Atención"] = dfMCL["Simulación"] * 50
    dfMCL

    buf = io.BytesIO()
    plt.plot(dfMCL['Simulación'], label='Simulación')
    plt.plot(dfMCL['Costo de Atención'], label='Costo de Atención')
    plt.legend()

    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data2 = dfMCL.to_html(
        classes="table table-ligth table-striped", justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/data.xlsx")
    dfMCL.to_excel(writer, index=False)
    writer.save()

    dfMCL.to_csv("static/file/data.csv", index=False)
 """
    return render_template('printSistemaMontecarlo.html', data=data, data2=data2, image=plot_url)


@app.route('/sistemaInventario')
def sistemaInventario():
    return render_template('sistemaInventario.html')


@app.route('/printSistemaInventario')
def imprimirSistemaInventario():
    return render_template('printSistemaInventario.html')


@app.route('/calcularSistemaInventario', methods=['GET', 'POST'])
def calcularSistemaInventario():
    D = request.form.get("demanda", type=float)
    Co = request.form.get("costoOrdenar", type=float)
    Ch = request.form.get("costoMantenimiento", type=float)
    P = request.form.get("costoProducto", type=float)
    Tespera = request.form.get("tiempoEspera", type=float)
    DiasAno = request.form.get("diasAno", type=int)
    num = request.form.get("numeroIteraciones", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    import math
    from math import sqrt
    from pandas import DataFrame

    Q = round(sqrt(((2*Co*D)/Ch)), 2)
    N = round(D / Q, 2)
    R = round((D / DiasAno) * Tespera, 2)
    T = round(DiasAno / N, 2)
    CoT = N * Co
    ChT = round(Q / 2 * Ch, 2)
    MOQ = round(CoT + ChT, 2)
    CTT = round(P * D + MOQ, 2)

    df = pd.DataFrame(columns=('Q', 'N', 'R', 'T', 'CoT', 'ChT', 'MOQ', 'CTT'))
    df.loc[len(df)] = [Q, N, R, T, CoT, ChT, MOQ, CTT]
    df

    data = df.to_html(classes="table table-striped",
                      justify="justify-all", border=0)

    # Programa para generar el gráfico de costo mínimo
    indice = ['Q', 'Costo_ordenar', 'Costo_Mantenimiento',
        'Costo_total', 'Diferencia_Costo_Total']
    # Generamos una lista ordenada de valores de Q

    periodo = np.arange(0, num)

    def genera_lista(Q):
        n = num
        Q_Lista = []
        i = 1
        Qi = Q
        Q_Lista.append(Qi)
        for i in range(1, 9):
            Qi = Qi - 60
            Q_Lista.append(Qi)

        Qi = Q
        for i in range(9, n):
            Qi = Qi + 60
            Q_Lista.append(Qi)
        return Q_Lista

    Lista = genera_lista(Q)
    Lista.sort()

    dfQ = DataFrame(index=periodo, columns=indice).fillna(0)

    dfQ['Q'] = Lista
    #dfQ

    for period in periodo:
        dfQ['Costo_ordenar'][period] = D * Co / dfQ['Q'][period]
        dfQ['Costo_Mantenimiento'][period] = dfQ['Q'][period] * Ch / 2
        dfQ['Costo_total'][period] = dfQ['Costo_ordenar'][period] + \
        dfQ['Costo_Mantenimiento'][period]
        dfQ['Diferencia_Costo_Total'][period] = dfQ['Costo_total'][period] - MOQ
    dfQ

    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(dfQ['Costo_ordenar'], label='Costo_ordenar')
    plt.plot(dfQ['Costo_Mantenimiento'], label='Costo_Mantenimiento')
    plt.plot(dfQ['Costo_total'], label='Costo_total')
    plt.legend()

    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    data2 = dfQ.to_html(classes="table table-ligth table-striped",
                        justify="justify-all", border=0)

    def make_data(product, policy, periods):
        periods += 1
        # Create zero-filled Dataframe
        period_lst = np.arange(periods)  # index
        header = ['INV_INICIAL', 'INV_NETO_INICIAL', 'DEMANDA', 'INV_FINAL', 'INV_FINAL_NETO',
            'VENTAS_PERDIDAS', 'INV_PROMEDIO', 'CANT_ORDENAR', 'TIEMPO_LLEGADA']
        df = DataFrame(index=period_lst, columns=header).fillna(0)
        # Create a list that will store each period order
        order_l = [Order(quantity=0, lead_time=0)
                   for x in range(periods)]
                       # Fill DataFrame
        for period in period_lst:
            if period == 0:
                df['INV_INICIAL'][period] = product.initial_inventory
                df['INV_NETO_INICIAL'][period] = product.initial_inventory
                df['INV_FINAL'][period] = product.initial_inventory
                df['INV_FINAL_NETO'][period] = product.initial_inventory
            if period >= 1:
                df['INV_INICIAL'][period] = df['INV_FINAL'][period - 1] + \
                    order_l[period - 1].quantity
                df['INV_NETO_INICIAL'][period] = df['INV_FINAL_NETO'][period -
                    1] + pending_order(order_l, period)
                #demand = int(product.demand())
                demand = D
                # We can't have negative demand
                if demand > 0:
                    df['DEMANDA'][period] = demand
                else:
                    df['DEMANDA'][period] = 0
                # We can't have negative INV_INICIAL
                if df['INV_INICIAL'][period] - df['DEMANDA'][period] < 0:
                    df['INV_FINAL'][period] = 0
                else:
                    df['INV_FINAL'][period] = df['INV_INICIAL'][period] - \
                        df['DEMANDA'][period]
                order_l[period].quantity, order_l[period].lead_time = placeorder(
                    product, df['INV_FINAL'][period], policy, period)
                df['INV_FINAL_NETO'][period] = df['INV_NETO_INICIAL'][period] - \
                    df['DEMANDA'][period]
                if df['INV_FINAL_NETO'][period] < 0:
                    df['VENTAS_PERDIDAS'][period] = abs(
                        df['INV_FINAL_NETO'][period])
                    df['INV_FINAL_NETO'][period] = 0
                else:
                    df['VENTAS_PERDIDAS'][period] = 0
                df['INV_PROMEDIO'][period] = (
                    df['INV_NETO_INICIAL'][period] + df['INV_FINAL_NETO'][period]) / 2.0
                df['CANT_ORDENAR'][period] = order_l[period].quantity
                df['TIEMPO_LLEGADA'][period] = order_l[period].lead_time
        return df

    def pending_order(order_l, period):
        """Return the order that arrives in actual period"""
        indices = [i for i, order in enumerate(order_l) if order.quantity]
        sum = 0
        for i in indices:
            if period-(i + order_l[i].lead_time+1) == 0:
                sum += order_l[i].quantity
        return sum

    def demanda(self):
            if self.demand_dist == "Constant":
                return self.demand_p1
            elif self.demand_dist == "Normal":
                return make_distribution(
                    np.random.normal,
                    self.demand_p1,
                    self.demand_p2)()
            elif self.demand_dist == "Triangular":
                return make_distribution(
                    np.random_triangular,
                    self.demand_p1,
                    self.demand_p2,
                    self.demand_p3)()
    def lead_time(self):
            if self.leadtime_dist == "Constant":
                return self.leadtime_p1
            elif self.leadtime_dist == "Normal":
                return make_distribution(
                    np.random.normal,
                    self.leadtime_p1,
                    self.leadtime_p2)()
            elif self.leadtime_dist == "Triangular":
                return make_distribution(
                    np.random_triangular,
                    self.leadtime_p1,
                    self.leadtime_p2,
                    self.leadtime_p3)()

    def __repr__(self):
           return '<Product %r>' % self.name

    def placeorder(product, final_inv_pos, policy, period):
        #lead_time = int(product.lead_time())
        lead_time = Tespera
        # Qs = if we hit the reorder point s, order Q units
        if policy['method'] == 'Qs' and \
                final_inv_pos <= policy['param2']:
            return policy['param1'], lead_time
        # RS = if we hit the review period R and the reorder point S, order: (S -
        # final inventory pos)
        elif policy['method'] == 'RS' and \
            period % policy['param1'] == 0 and \
                final_inv_pos <= policy['param2']:
            return policy['param2'] - final_inv_pos, lead_time
        else:
            return 0, 0

    politica = {'method': "Qs", 'param1': 50,'param2': 20}

    class Order(object):
        """Object that stores basic data of an order"""

        def __init__(self, quantity, lead_time):
            self.quantity = quantity
            self.lead_time = lead_time

    class product(object):
        def __init__ (self, name,price,order_cost,initial_inventory,demand_dist,demand_p1,demand_p2,demand_p3,leadtime_dist,leadtime_p1,leadtime_p2,leadtime_p3):
            self.name = name
            self.price = price
            self.order_cost = order_cost
            self.initial_inventory = initial_inventory
            self.demand_dist = demand_dist
            self.demand_p1 = demand_p1
            self.demand_p2 = demand_p2
            self.demand_p3 = demand_p3
            self.leadtime_dist = leadtime_dist
            self.leadtime_p1 = leadtime_p1
            self.leadtime_p2 = leadtime_p2
            self.leadtime_p3 = leadtime_p3
    producto = product("Mesa", 18.0, 20.0,100,"Constant",80.0,0.0,0.0,"Constant",1.0,0.0,0.0)

    num = num - 1
    df = make_data(producto, politica, num)
    df

    data3 = df.to_html(classes="table table-ligth table-striped",
                       justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/autos.xlsx")
    df.to_excel(writer, index=False)
    writer.save()

    df.to_csv("static/file/data.csv", index=False) """

    return render_template('printSistemaInventario.html', data=data, data2=data2, data3=data3, image=plot_url)


@app.route('/sistemaLineaEspera')
def sistemaLineaEspera():
    return render_template('sistemaLineaEspera.html')


@app.route('/printLineaEspera')
def imprimirLineaEspera():
    return render_template('printLineaEspera.html')


@app.route('/calcularLineaEspera', methods=['GET', 'POST'])
def calcularLineaEspera():
    landa = request.form.get("landa", type=float)
    nu = request.form.get("miu", type=float)
    num = request.form.get("numeroIteraciones", type=int)
    x0 = request.form.get("semilla", type=int)
    a = request.form.get("multiplicador", type=int)
    c = request.form.get("incremento", type=int)
    m = request.form.get("modulo", type=int)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas import ExcelWriter
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    from io import BytesIO
    import base64
    import math
    import random
    from pandas import DataFrame

    #La probabilidad de hallar el sistema ocupado o utilización del sistema:
    p = []
    p = landa/nu
    #La probabilidad de que no haya unidades en el sistema este vacía u ocioso :
    Po = []
    Po = 1.0 - (landa/nu)
    #Longitud esperada en cola, promedio de unidades en la línea de espera:
    Lq = []
    Lq = landa*landa / (nu * (nu - landa))
    #/ (nu * (nu - landa))
    # Número esperado de clientes en el sistema(cola y servicio) :
    L = []
    L = landa / (nu - landa)
    #El tiempo promedio que una unidad pasa en el sistema:
    W = []
    W = 1 / (nu - landa)
    #Tiempo de espera en cola:
    Wq = []
    Wq = W - (1.0 / nu)
    print(Wq)
    #La probabilidad de que haya n unidades en el sistema:
    n = 1
    Pn = []
    Pn = (landa/nu)*n*Po

    df = pd.DataFrame(columns=('lambda', 'nu', 'p',
                      'Po', 'Lq', 'L', 'W', 'Wq', 'Pn'))
    df.loc[len(df)] = [landa, nu, p, Po, Lq, L, W, Wq, Pn]
    df

    data = df.to_html(classes="table table-ligth table-striped",
                      justify="justify-all", border=0)

    i = 0
    # Landa y nu ya definidos
    # Atributos del DataFrame
    """
    ALL # ALEATORIO DE LLEGADA DE CLIENTES
    ASE # ALEATORIO DE SERVICIO
    TILL TIEMPO ENTRE LLEGADA
    TISE TIEMPO DE SERVICIO
    TIRLL TIEMPO REAL DE LLEGADA
    TIISE TIEMPO DE INICIO DE SERVICIO
    TIFSE TIEMPO FINAL DE SERVICIO
    TIESP TIEMPO DE ESPERA
    TIESA TIEMPO DE SALIDA
    numClientes NUMERO DE CLIENTES
    dfLE DATAFRAME DE LA LINEA DE ESPERA
    """
    numClientes = num
    i = 0
    indice = ['ALL', 'ASE', 'TILL', 'TISE',
              'TIRLL', 'TIISE', 'TIFSE', 'TIESP', 'TIESA']
    Clientes = np.arange(numClientes)
    dfLE = pd.DataFrame(index=Clientes, columns=indice).fillna(0.000)

    #np.random.seed(num)

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * num
    r = [0.1] * num
    for j in range(0, num):
        x[j] = ((a*x0)+c) % m
        x0 = x[j]
        #r[j] = x0 / m
        dfLE['ALL'][j] = x0 / m
        #dfLE['ASE'][j] = x0 / m

    # n, m, a, x0 = 20, 1000, 747, 123
    x = [1] * num
    r = [0.1] * num
    for j in range(0, num):
        x[j] = (a*x0) % m
        x0 = x[j]
        #r[j] = x0 / m
        #dfLE['ALL'][j] = x0 / m
        dfLE['ASE'][j] = x0 / m

    for i in Clientes:
        if i == 0:
            #dfLE['ASE'][i] = random.random()
            dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
            dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
            dfLE['TIRLL'][i] = dfLE['TILL'][i]
            dfLE['TIISE'][i] = dfLE['TIRLL'][i]
            dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
            dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
        else:
            #dfLE['ASE'][i] = random.random()
            dfLE['TILL'][i] = -landa*np.log(dfLE['ALL'][i])
            dfLE['TISE'][i] = -nu*np.log(dfLE['ASE'][i])
            dfLE['TIRLL'][i] = dfLE['TILL'][i] + dfLE['TIRLL'][i-1]
            dfLE['TIISE'][i] = max(dfLE['TIRLL'][i], dfLE['TIFSE'][i-1])
            dfLE['TIFSE'][i] = dfLE['TIISE'][i] + dfLE['TISE'][i]
            dfLE['TIESP'][i] = dfLE['TIISE'][i] - dfLE['TIRLL'][i]
            dfLE['TIESA'][i] = dfLE['TIESP'][i] + dfLE['TISE'][i]
    nuevas_columnas = pd.core.indexes.base.Index(["A_LLEGADA", "A_SERVICIO", "TIE_LLEGADA", "TIE_SERVICIO",
                                                  "TIE_EXACTO_LLEGADA", "TIE_INI_SERVICIO", "TIE_FIN_SERVICIO",
                                                  "TIE_ESPERA", "TIE_EN_SISTEMA"])

    dfLE.columns = nuevas_columnas
    dfLE

    # Graficamos los numeros generados
    buf = io.BytesIO()
    plt.plot(dfLE['A_LLEGADA'], label='A_LLEGADA')
    plt.plot(dfLE['A_SERVICIO'], label='A_SERVICIO')
    plt.plot(dfLE['TIE_LLEGADA'], label='TIE_LLEGADA')
    plt.plot(dfLE['TIE_SERVICIO'], label='TIE_SERVICIO')
    plt.plot(dfLE['TIE_EXACTO_LLEGADA'], label='TIE_EXACTO_LLEGADA')
    plt.plot(dfLE['TIE_INI_SERVICIO'], label='TIE_INI_SERVICIO')
    plt.plot(dfLE['TIE_FIN_SERVICIO'], label='TIE_FIN_SERVICIO')
    plt.plot(dfLE['TIE_ESPERA'], label='TIE_ESPERA')
    plt.plot(dfLE['TIE_EN_SISTEMA'], label='TIE_EN_SISTEMA')
    plt.legend(loc=2)
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    plot_url = base64.b64encode(buf.getvalue()).decode('UTF-8')

    kl = dfLE["TIE_ESPERA"]
    jl = dfLE["TIE_EN_SISTEMA"]
    ll = dfLE["A_LLEGADA"]
    pl = dfLE["A_SERVICIO"]
    ml = dfLE["TIE_INI_SERVICIO"]
    nl = dfLE["TIE_FIN_SERVICIO"]

    klsuma = sum(kl)
    klpro = (klsuma/num)
    jlsuma = sum(jl)
    jlpro = jlsuma/num
    dfLE.loc[num] = ['-', '-', '-', '-', '-', '-', 'SUMA', klsuma, jlsuma]
    dfLE.loc[(num+1)] = ['-', '-', '-', '-',
                         '-', '-', 'PROMEDIO', klpro, jlpro]

    dfLE

    data2 = dfLE.to_html(
        classes="table table-ligth table-striped", justify="justify-all", border=0)

    """ writer = ExcelWriter("static/file/autos.xlsx")
    dfLE.to_excel(writer, index=False)
    writer.save()

    dfLE.to_csv("static/file/data.csv", index=False) """

    dfLE2 = pd.DataFrame(dfLE.describe())
    data3 = dfLE2.to_html(
        classes="table table-ligth table-striped", justify="justify-all", border=0)

    return render_template('printLineaEspera.html', data=data, data2=data2, data3=data3, image=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
