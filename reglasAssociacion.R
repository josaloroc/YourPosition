#Libraries
install.packages('caret', dependencies = TRUE)
install.packages('devtools')
install.packages('tidyverse', dependencies = TRUE)
install.packages('parallel')
install.packages('doParallel')
install.packages("corrplot")
install.packages("gridExtra")
install.packages("GGally")
install.packages("knitr")
install.packages("arules")
install.packages("arulesViz")
install.packages("RColorBrewer")
library(caret)
library(tidyverse)
library(parallel)
library(doParallel)
library(corrplot)
library(gridExtra)
library(GGally)
library(knitr)
library(arules)
library(arulesViz)
library(RColorBrewer)

#Read data
data <- read.csv2("data.csv", dec = ".")
data$Pos <- as.factor(data$Pos)
data <- data %>% select(-c(Rk, Player, Nation, Squad, Comp, Age, Born))
data <- subset(data, MP>10)
data <- data %>% select(-c(MP,Starts,Min,X90s))
data <- droplevels(data)
dim(data)
class(data)

lapply(data, class)

summary(data)

# CONVERSIÓN DE UN DATAFRAME A UN OBJETO TIPO TRANSACTION
# ==============================================================================
# Se convierte el dataframe a una lista en la que cada elemento  contiene los
# items de una transacción
datos_split <- split(x = data$Pos, f = data$Shots)
transacciones <- as(datos_split, Class = "transactions")
transacciones


colnames(transacciones)[1:10]

rownames(transacciones)[1:366]

inspect(transacciones[1:366])

#También es posible mostrar los resultados en formato de dataframe con la función DATAFRAME() o con as(transacciones, "dataframe").
df_transacciones <- as(transacciones, Class = "data.frame")
# Para que el tamaño de la tabla se ajuste mejor, se convierte el dataframe a tibble
as_tibble(df_transacciones) %>% head()

#Para extraer el tamaño de cada transacción se emplea la función size().
tamanyos <- size(transacciones)
summary(tamanyos)
quantile(tamanyos, probs = seq(0,1,0.1))

data.frame(tamanyos) %>%
  ggplot(aes(x = tamanyos)) +
  geom_histogram() +
  labs(title = "Distribución del tamaño de las transacciones",
       x = "Tamaño") +
  theme_bw()

frecuencia_items <- itemFrequency(x = transacciones, type = "relative")
frecuencia_items %>% sort(decreasing = TRUE) %>% head(10)

frecuencia_items <- itemFrequency(x = transacciones, type = "absolute")
frecuencia_items %>% sort(decreasing = TRUE) %>% head(10)

soporte <- 30 / dim(transacciones)[1]
itemsets <- apriori(data = transacciones,
                    parameter = list(support = soporte,
                                     minlen = 1,
                                     maxlen = 20,
                                     target = "frequent itemset"))

summary(itemsets)

# Se muestran los top total itemsets de mayor a menor soporte
top_20_itemsets <- sort(itemsets, by = "support", decreasing = TRUE)
inspect(top_20_itemsets)

# Para representarlos con ggplot se convierte a dataframe 
as(top_20_itemsets, Class = "data.frame") %>%
  ggplot(aes(x = reorder(items, support), y = support)) +
  geom_col() +
  coord_flip() +
  labs(title = "Itemsets más frecuentes", x = "itemsets") +
  theme_bw()

# Se muestran los itemsets más frecuentes formados por más de un item.
inspect(sort(itemsets[size(itemsets) > 1], decreasing = TRUE))

soporte <- 30 / dim(transacciones)[1]
reglas <- apriori(data = transacciones,
                  parameter = list(support = soporte,
                                   confidence = 0.70,
                                   # Se especifica que se creen reglas
                                   target = "rules"))
summary(reglas)

inspect(sort(x = reglas, decreasing = TRUE, by = "confidence"))

metricas <- interestMeasure(reglas, measure = c("coverage", "fishersExactTest"),
                            transactions = transacciones)
metricas

quality(reglas) <- cbind(quality(reglas), metricas)
inspect(sort(x = reglas, decreasing = TRUE, by = "confidence"))
df_reglas <- as(reglas, Class = "data.frame") 
df_reglas %>% as.tibble() %>% arrange(desc(confidence)) %>% head()

soporte <- 30 / dim(transacciones)[1]
reglas_ab <- apriori(data = transacciones,
                             parameter = list(support = soporte,
                                              confidence = 0.70,
                                              # Se especifica que se creen reglas
                                              target = "rules"))
summary(reglas_ab)


reglas_maximales <- reglas[is.maximal(reglas)]
reglas_maximales
inspect(reglas_maximales)

reglas_redundantes <- reglas[is.redundant(x = reglas, measure = "confidence")]
reglas_redundantes

# Se identifica la regla con mayor confianza
as(reglas, "data.frame") %>%
  arrange(desc(confidence)) %>%
  head(1) %>%
  pull(rules)

filtrado_transacciones <- subset(x = transacciones,
                                 subset = items %ain% c("DF", "MFFW",
                                                        "MF"))
filtrado_transacciones
inspect(filtrado_transacciones[1:3])
