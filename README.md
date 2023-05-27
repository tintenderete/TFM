# TFM

Algoritmos con redes evolutivas
Idea de proyecto:

La combinación de redes neuronales y algoritmos genéticos no es nuevo. Sin embargo el plantearse la existencia de una red con topología evolutiva sí que lo es. 
Hasta ahora, las redes neuronales tenían dos fases: Entrenamiento y producción. Una vez optimizado el entrenamiento, se había llegado a una conclusión sobre la 
topología de la red (número de capas, número de neuronas en cada capa, función de activación… etc). 

En un entrenamiento online, la red está permanentemente aprendiendo de los nuevos inputs que le llegan. Adaptándose a través de la modificación de sus pesos. 
Pero la topología permanece constante en el tiempo. Aquí es donde empieza este trabajo.  

El objetivo de este TFM es diseñar un algoritmo de inversión (no tiene porqué ser muy complejo) que utilice una red de topología evolutiva, mediante algoritmos genéticos.
Para analizar la eficiencia de la evolución, deberemos construir dos redes neuronales, que optimicen el mismo algoritmo de inversión. 
El objetivo es comparar los resultados que obtiene el algoritmo de inversión con la primera red, una red no evolutiva, con una red cuya topología podría evolucionar 
con cada input que recibe. 

El objetivo es diseñar una red que evolucione con el tiempo, adaptándose al problema a través de su propia modificación. 
