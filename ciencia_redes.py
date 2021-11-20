def muestraInfoBasica(file):
  G = nx.read_edgelist(file, comments='#',
                     create_using=nx.Graph(), 
                     delimiter=' ', 
                     nodetype=int, 
                     encoding='utf-8')
  N = len(G)
  L = G.size()
  degrees = list(dict(G.degree()).values())
  kmin = min(degrees)
  kmax = max(degrees)
  print("Número de nodos: ", N)
  print("Número de enlaces: ", L)
  print('-------')
  print("Grado promedio: ", 2*L/N) #Formula vista en clases (qué sucedía con las redes reales?)
  print("Grado promedio (alternativa de calculo)", np.mean(degrees))
  print('-------')
  print("Grado mínimo: ", kmin)
  print("Grado máximo: ", kmax)
  return G

def graficarDistribucion(G, escala_lineal = False, n_examples = 20):
  degrees = list(dict(G.degree()).values())
  kmin = min(degrees)
  kmax = max(degrees) 
  # Generamos 10 bins espaciados logaritmicamente entre kmin y kmax
  
  if escala_lineal == False:
    bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), num=n_examples)

    # histograma de la data para esos bines
    density, _ = np.histogram(degrees, bins=bin_edges, density=True)
    log_be = np.log10(bin_edges)
    x = 10**((log_be[1:] + log_be[:-1])/2)

    plt.loglog(x, density, marker='o', linestyle='none')
    plt.xlabel(r"degree $k$", fontsize=16)
    plt.ylabel(r"$P(k)$", fontsize=16)
  else:
    # Entrega 20 bins linealmente espaceados entre kmin y kmax
    bin_edges = np.linspace(kmin, kmax, num=n_examples)

    # histograma de la data en estos bines 
    density, _ = np.histogram(degrees, bins=bin_edges, density=True)

    # "x" debería ser el punto meido (en escala lineal) de cada bin
    log_be = np.log10(bin_edges)
    
    #log_be = bin_edges
    x = 10**((log_be[1:] + log_be[:-1])/2)
    
    plt.plot(x, density, marker='o', linestyle='none')
    plt.xlabel(r"degree $k$", fontsize=16)
    plt.ylabel(r"$P(k)$", fontsize=16)
  
  fig = plt.figure(figsize=(6,4))

  # "x" debe ser el punto medio (en escala LOG) de cada bin

  # # remuevo los limites derecho y superior 
  # ax = plt.gca()
  # ax.spines['right'].set_visible(False)
  # ax.spines['top'].set_visible(False)
  # ax.yaxis.set_ticks_position('left')
  # ax.xaxis.set_ticks_position('bottom')

  # Muestra la gráfica
  plt.show()
  
  
  
  
  
