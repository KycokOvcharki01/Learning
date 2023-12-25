library(readxl)
library(corrplot)
library(stargazer)
library(ggplot2)
library(dplyr)
library(plm)
library(lmtest)
library(car)
# функция скорректированных ошибок ----------------------------------------

clse = function(reg) { 
  # index(reg, "id") returns the id or entity variable vector 
  G = length(unique(index(reg,"id")))
  N = length(index(reg,"id"))
  dfa = (G/(G - 1))   # note Bluhm multiplies this by finite-sample df adjustment
  rob = sqrt(diag(dfa*vcovHC(reg, method="arellano", type = "HC1", 
                             cluster = "group")))
  return(rob)
}

cse = function(reg) {
  rob = sqrt(diag(vcovHC(reg, type = "HC1")))
  return(rob)
}

# Импорт данных -----------------------------------------------------------


# Загрузим данные с интерполяцией и дисагрегацией (сложные)
link1 <- 'data_disag_interpol_since_2020_12.xlsx'
data_complicated <- read_xlsx(link1)

# Загрузим чистые данные (простые)
link2 <- 'fin_data_no_ful_vac.xlsx'
data_simple <- read_xlsx(link2)


# Добавим уровни вакцинации и посчитаем приросты*100 -----------------------------------------------

data_complicated$vac_lvl <- data_complicated$people_vaccinated/data_complicated$population*100
data_simple$vac_lvl <- data_simple$people_vaccinated/data_simple$population*100


# прирост вакцинируемых для простых данных
data_dif_simple <- data.frame()
for (i in unique(data_simple$iso_code)) {
  data_ord <- subset(data_simple, iso_code == i) %>% arrange(date) %>% mutate(dif_vac = vac_lvl - dplyr::lag(vac_lvl))
  data_dif_simple <- rbind(data_dif_simple,data_ord[2:nrow(data_ord),])
}
# прирост вакцинируемых для сложных данных
data_dif_complicated <- data.frame()
for (i in unique(data_complicated$iso_code)) {
  data_ord <- subset(data_complicated, iso_code == i) %>% arrange(date) %>% mutate(dif_vac = vac_lvl - dplyr::lag(vac_lvl))
  data_dif_complicated <- rbind(data_dif_complicated,data_ord[1:nrow(data_ord)-1,])
}


# сделаем страны факторной переменной -------------------------------------
data_dif_simple$iso_code <- as.factor(data_dif_simple$iso_code)
data_dif_complicated$iso_code <- as.factor(data_dif_complicated$iso_code)
str(data_dif_complicated)
str(data_dif_simple)


# Количество наблюдений для каждой страны ---------------------------------

obs_num <- data_dif_simple %>% group_by(location) %>% dplyr::summarise(n = n() )
obs_num2 <- data_dif_complicated %>% group_by(location) %>% dplyr::summarise(n = n() )
data_obs_num <- as.matrix(merge(obs_num,obs_num2, by = 'location'))
colnames(data_obs_num) <- c('location', 'simple', 'complicated')
stargazer(data_obs_num, type="text")

# спецификация ------------------------------------------------------------

#  Построим регресии lm и попытаемся выбрать подходящую спецификацию

mod1 <- lm(dif_vac ~ log(gdp_per_capita)+ stringency_index+ human_development_index+ 
             CPI+ score_government+democracy_lvl,
           data = data_dif_complicated)
summary(mod1)

# рассмотрим предположение что прирост вакцинацинируемых имеет квадратичный вид
plot(mod1, which = 1)

# посмотрим на остатки
plot(mod1, which = 2)

# Посмотрим на график где Размер кружка определяется расстоянием Кука, а по осям
#  отложены остатки и hat values
influencePlot(mod1)
#  выбросы присутствуют
# посмотрим на мультиколлинеарность (сравниваем с 10)
vif(mod1)
# смело выбросим ввп
mod2 <- update(mod1, .~.-log(gdp_per_capita))
summary(mod2)
# посмотрим сново на мультиколрринеарность, все стало лучше
vif(mod2)
# а что скажет тест короткая против длинной?
anova(mod1, mod2)
# на 1% уровне короткая лучше, это хорошо
# рассмотрим предположение что прирост вакцинацинируемых имеет квадратичный вид
plot(mod2, which = 1)

# посмотрим на остатки
plot(mod2, which = 2)
#  Подозрения вызывают 211 и 210 наблюдения
data_dif_complicated[210,]
data_dif_complicated[211,]
# это наблюдения по Сингапуру за 2020-12-31 и 2021-01-31
# что если их выкинуть?
# выкидываем Сингапур целиком, чтобы сохранить сбалансированность панели
data_dif_complicated_update <- subset(data_dif_complicated, location != 'Singapore')
# построим модель заново
mod3 <- lm(dif_vac ~  stringency_index+ human_development_index+ 
             CPI+ score_government+democracy_lvl,
           data = data_dif_complicated_update)
summary(mod3)
# рассмотрим предположение что прирост вакцинацинируемых имеет квадратичный вид
plot(mod3, which = 1)

# посмотрим на остатки
plot(mod3, which = 2)
# с графиком influence plot все стало значительно лучше
influencePlot(mod3)
# Посмотрим на графики остатков
crPlots(mod3)
# ничего по спецификации добавлять не надо
# а тест рамсея?
resettest(mod3, power = 2)
# пропустили квадраты
resettest(mod3, power = 3)
# пропустили кубы, но тогда это сложно интерпертировать
# Попробуем добавить степени, больше всего похоже что стоит добавить квадрат 
# доверия к государтсву
mod4 <- update(mod3, .~.+I(score_government^2))
summary(mod4)
# не похоже что это государство
# может stringency_index?
mod5 <- update(mod3, .~.+I(stringency_index^2))
summary(mod5)
# тоже не то, может CPI?
mod6 <- update(mod3, .~.+I(CPI^2))
summary(mod6)
# тоже нет, может democracy_lvl
mod7 <- update(mod3, .~.+I(democracy_lvl^2))
summary(mod7)
# Нашли! что с тестом Рамсея?
resettest(mod7, power = 2)
# принимаем H0: степени не пропущены!
# короткая против длинной
anova(mod7, mod3)
# длинная лучше
# посмотрим на остатки,хаос- хорошо
plot(mod7, which = 1)
# а crPlots?
crPlots(mod7)
# ничего не стоит менять
# Со спецификацией вроде разобрались



# количество наблюдений после спецификации --------------------------------

obs_num <- data_dif_simple %>% group_by(location) %>% dplyr::summarise(n = n() )
obs_num2 <- data_dif_complicated_update %>% group_by(location) %>% dplyr::summarise(n = n() )
obs_num2 <- rbind(obs_num2, c('Singapore', 0))
data_obs_num <- as.matrix(merge(obs_num,obs_num2, by = 'location'))
colnames(data_obs_num) <- c('location', 'simple', 'complicated')
stargazer(data_obs_num, type="text")



# Построение моделей где спецификация с квадратом ------------------------------------------------------

# Регрессия пула простые данные
reg_pool_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
               CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
             data = data_dif_simple, index = c("iso_code","date"), model="pooling")
summary(reg_pool_1)

# Регрессия пула сложные данные
reg_pool_2 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                     CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                   data = data_dif_complicated_update, index = c("iso_code","date"), model="pooling")
summary(reg_pool_2)

# Сравним модели
stargazer(reg_pool_1, reg_pool_2, se = list(clse(reg_pool_1), clse(reg_pool_2)),
          type = 'text', column.labels = c('Simple Pool', 'Complicated Pool'))

# Сравним обычную регрессию и регресиию с фиктивными времени
# простые данные без фиктивных переменных
reg_lm_1_q_1 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                 data = data_dif_simple)
summary(reg_lm_1_q_1)
# сложные данные без фиктивных переменных
reg_lm_2_q_1 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                 data = data_dif_complicated_update)
summary(reg_lm_2_q_1)


# простые данные С фиктивных переменных
reg_lm_1_q_2 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+I(democracy_lvl^2)+date,
                 data = data_dif_simple)
summary(reg_lm_1_q_2)
# сложные данные С фиктивных переменных
reg_lm_2_q_2 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+I(democracy_lvl^2)+date,
                 data = data_dif_complicated_update)
summary(reg_lm_2_q_2)

# короткая против длинной
anova(reg_lm_1_q_1,reg_lm_1_q_2)
anova(reg_lm_2_q_1,reg_lm_2_q_2)


# Модель с фиксированными эффектами простые данные
reg_within_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                       CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
             data = data_dif_simple, index = c("iso_code","date"), model="within", effect="individual")
summary(reg_within_1)

# Модель с фиксированными эффектами сложные данные
reg_within_2 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                       CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                     data = data_dif_complicated_update, index = c("iso_code","date"), model="within", effect="individual")
summary(reg_within_2)

# Сравним модели
stargazer(reg_within_1, reg_within_2, se = list(clse(reg_within_1), clse(reg_within_2)),
          type = 'text', column.labels = c('Simple Within', 'Complicated Within'))


# Модель с фиктивными переменными времени простые данные

reg_within_time_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                            CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
             data = data_dif_simple, index = c("iso_code","date"), model="within", effect="twoways")
summary(reg_within_time_1)

# Модель с фиктивными переменными времени сложные данные
reg_within_time_2 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                            CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                          data = data_dif_complicated_update, index = c("iso_code","date"), model="within", effect="twoways")
summary(reg_within_time_2)

# табличка
stargazer(reg_within_time_1, reg_within_time_2, se = list(clse(reg_within_time_1), clse(reg_within_time_2)),
          type = 'text', column.labels = c('Simple Within', 'Complicated Within'))


# Сравниваем регрессию пула и FE двунаправленная
# H0: регрессия пула лучше
plmtest(reg_pool_1,effect="twoways",type="ghm")
plmtest(reg_pool_2,effect="twoways",type="ghm")

# Сравниваем регрессию пула и FE 
# H0: регрессия пула лучше
plmtest(reg_pool_1,effect="individual",type="honda")
plmtest(reg_pool_2,effect="individual",type="honda")

# Сравниваем регрессию пула и RE
# H0: регрессия пула лучше
plmtest(reg_pool_1,effect="twoway",type="bp")
plmtest(reg_pool_2,effect="twoway",type="bp")

# Модель в первых разностях
reg_dif_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                    CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
             data = data_dif_simple, index = c("iso_code","date"),  model="fd")
summary(reg_dif_1)

reg_dif_2 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                    CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                  data = data_dif_complicated_update, index = c("iso_code","date"),  model="fd")
summary(reg_dif_2)

stargazer(reg_dif_1, reg_dif_2, se = list(clse(reg_dif_1), clse(reg_dif_2)),
          type = 'text', column.labels = c('Simple FD', 'Complicated FD'))


# модель со случайными эффектами

reg_RE_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                  data = data_dif_simple, index = c("iso_code","date"), effect="twoway", model="random")
summary(reg_RE_1)



reg_RE_2 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+I(democracy_lvl^2),
                  data = data_dif_complicated_update, index = c("iso_code","date"),effect="twoway",  model="random")
summary(reg_RE_2)

stargazer(reg_RE_1, reg_RE_2, se = list(clse(reg_RE_1), clse(reg_RE_2)),
          type = 'text', column.labels = c('Simple RE', 'Complicated RE'))

#Тест Хаусмана для сравнения FE и RE моделей
#H0: выбирае RE
phtest(reg_within_1, reg_RE_1)
phtest(reg_within_2, reg_RE_2)



# построение моделей,  где спецификация линейная ----------------------


# Регрессия пула простые данные
reg_pool_1_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                     CPI+ score_government+democracy_lvl,
                   data = data_dif_simple, index = c("iso_code","date"), model="pooling")
summary(reg_pool_1_1)

# Регрессия пула сложные данные
reg_pool_2_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                     CPI+ score_government+democracy_lvl,
                   data = data_dif_complicated_update, index = c("iso_code","date"), model="pooling")
summary(reg_pool_2_1)

# Сравним обычную регрессию и регресиию с фиктивными времени
# простые данные без фиктивных переменных
reg_lm_1_1 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
             CPI+ score_government+democracy_lvl,
           data = data_dif_simple)
summary(reg_lm_1_1)
# сложные данные без фиктивных переменных
reg_lm_2_1 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl,
                 data = data_dif_complicated_update)
summary(reg_lm_2_1)


# простые данные С фиктивных переменных
reg_lm_1_2 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+date,
                 data = data_dif_simple)
summary(reg_lm_1_2)
# сложные данные С фиктивных переменных
reg_lm_2_2 <- lm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl+date,
                 data = data_dif_complicated_update)
summary(reg_lm_2_2)

# короткая против длинной
anova(reg_lm_1_1,reg_lm_1_2)
anova(reg_lm_2_1,reg_lm_2_2)

# Сравним модели
stargazer(reg_pool_1_1, reg_pool_2_1, se = list(clse(reg_pool_1_1), clse(reg_pool_2_1)),
          type = 'text', column.labels = c('Simple Pool', 'Complicated Pool'))

# Модель с фиксированными эффектами простые данные
reg_within_1_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                       CPI+ score_government+democracy_lvl,
                     data = data_dif_simple, index = c("iso_code","date"), model="within", effect="individual")
summary(reg_within_1_1)

# Модель с фиксированными эффектами сложные данные
reg_within_2_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                       CPI+ score_government+democracy_lvl,
                     data = data_dif_complicated_update, index = c("iso_code","date"), model="within", effect="individual")
summary(reg_within_2_1)

# Сравним модели
stargazer(reg_within_1_1, reg_within_2_1, se = list(clse(reg_within_1_1), clse(reg_within_2_1)),
          type = 'text', column.labels = c('Simple Within', 'Complicated Within'))


# Модель с фиктивными переменными времени простые данные

reg_within_time_1_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                            CPI+ score_government+democracy_lvl,
                          data = data_dif_simple, index = c("iso_code","date"), model="within", effect="twoways")
summary(reg_within_time_1_1)

# Модель с фиктивными переменными времени сложные данные
reg_within_time_2_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                            CPI+ score_government+democracy_lvl,
                          data = data_dif_complicated_update, index = c("iso_code","date"), model="within", effect="twoways")
summary(reg_within_time_2_1)

# табличка
stargazer(reg_within_time_1_1, reg_within_time_2_1, se = list(clse(reg_within_time_1_1), clse(reg_within_time_2_1)),
          type = 'text', column.labels = c('Simple Within', 'Complicated Within'))


# Сравниваем регрессию пула и FE двунаправленная
# H0: регрессия пула лучше
plmtest(reg_pool_1_1,effect="twoways",type="ghm")
plmtest(reg_pool_2_1,effect="twoways",type="ghm")
# Сравниваем регрессию пула и FE 
# H0: регрессия пула лучше
plmtest(reg_pool_1_1,effect="individual",type="honda")
plmtest(reg_pool_2_1,effect="individual",type="honda")

# Сравниваем регрессию пула и RE
# H0: регрессия пула лучше
plmtest(reg_pool_1_1,effect="twoway",type="bp")
plmtest(reg_pool_2_1,effect="twoway",type="bp")

# Модель в первых разностях
reg_dif_1_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                    CPI+ score_government+democracy_lvl,
                  data = data_dif_simple, index = c("iso_code","date"),  model="fd")
summary(reg_dif_1_1)

reg_dif_2_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                    CPI+ score_government+democracy_lvl,
                  data = data_dif_complicated_update, index = c("iso_code","date"),  model="fd")
summary(reg_dif_2_1)

stargazer(reg_dif_1_1, reg_dif_2_1, se = list(clse(reg_dif_1_1), clse(reg_dif_2_1)),
          type = 'text', column.labels = c('Simple FD', 'Complicated FD'))


# модель со случайными эффектами

reg_RE_1_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl,
                 data = data_dif_simple, index = c("iso_code","date"), effect="twoway", model="random")
summary(reg_RE_1_1)



reg_RE_2_1 <-  plm(dif_vac ~ stringency_index+ human_development_index+ 
                   CPI+ score_government+democracy_lvl,
                 data = data_dif_complicated_update, index = c("iso_code","date"),effect="twoway",  model="random")
summary(reg_RE_2_1)

stargazer(reg_RE_1_1, reg_RE_2_1, se = list(clse(reg_RE_1_1), clse(reg_RE_2_1)),
          type = 'text', column.labels = c('Simple RE', 'Complicated RE'))

#Тест Хаусмана для сравнения FE и RE моделей
#H0: выбирае RE
phtest(reg_within_1_1, reg_RE_1_1)
phtest(reg_within_2_1, reg_RE_2_1)


# сравнение обычных МНК ---------------------------------------------------

# тест коротокая против длинной на добавление квадрата для простых данных
anova(reg_lm_1_1, reg_lm_1_q_1)

# тест коротокая против длинной на добавление квадрата для сложных данных
anova(reg_lm_2_1, reg_lm_2_q_1)


# тест влияния временных эффетов на FE модели -----------------------------
# для простых данных
plmtest(reg_within_time_1_1)
plmtest(reg_within_time_1)

# для сложных данных
plmtest(reg_within_time_2_1)
plmtest(reg_within_time_2)

# Построение графиков и описательных статистик ----------------------------
data <- data_simple
data_dif <- data_dif_simple


# строим корреляционную матрицу, внутри просто выбираются нужные столбцы датасета
matr_2 <- cor(na.omit(data_dif[,6:16]) %>% dplyr::select(-vac_lvl))
# корреляционная матрица 9 на 9 не очень смотрится в виде таблицы  
# конкретные значения нам не особо нужны,
# поэтому построим визуализацию
corrplot(matr_2, method = 'pie')

# чб для презы
corrplot(matr_2, method = 'circle', col=colorRampPalette(c("white","grey","black"))(200))

plot(matr_2)


# график распределиния стран по уровню доверия к государству и уровню демократии
# достаем данные. Так как оба показателя годовые, то можно взять все нужные столбцы
# и выбрать уникальные строки, чтобы не было повторений 
data_plot_1 <- subset(data, as.numeric(format(as.Date(data$date),"%Y")) == 2021) %>%
  select(location, score_government, democracy_lvl) %>% unique()

# строим сам график
ggplot(data = data_plot_1, 
       aes(x = score_government, y = democracy_lvl,
           color = factor(location))) +
  geom_point(alpha = 0.3) + geom_text(label = data_plot_1$location, check_overlap = T) +
  annotate("rect", xmin=c(32,19,52,76), xmax=c(63,47,80, 95), 
           ymin=c(7.5,4.5, 5.9,1.9) , ymax=c(9.2, 7.4, 7.4,3.1) 
           , alpha=0.1, color="black", fill=c("blue", 'orange', 'purple', 'green'))+
  theme(legend.position="none")+
  ggtitle('Clustering by democracy level and government trust')


# Чб версия
ggplot(data = data_plot_1, 
       aes(x = score_government, y = democracy_lvl)) +
  geom_point(alpha = 0.3) + geom_text(label = data_plot_1$location, check_overlap = T) +
  annotate("rect", xmin=c(32,19,52,76), xmax=c(63,47,80, 95), 
           ymin=c(7.5,4.5, 5.9,1.9) , ymax=c(9.2, 7.4, 7.4,3.1) 
           , alpha=0.1, color="black")+
  theme(legend.position="none")+
  ggtitle('Clustering by democracy level and government trust')



# график распределиния стран по уровню доверия к государству и уровню вакцинации
# выбираем нужные столбцы. берем последние доступные сведеньи об уровне вакцинации

data_plot_2 <- subset(data, (as.numeric(format(as.Date(data$date),"%Y")) == 2021) &
                        as.numeric(format(as.Date(data$date),"%m")) == 12) %>% 
  select(location, score_government, vac_lvl)

# добавим отдельно ОАЭ так как только у этой страны отсутствуют данные о вакцинации за декабрь 2021
data_plot_2 <- rbind(data_plot_2, subset(data, (iso_code == 'ARE') & (date == '2021-11-25') ) %>%
                       select(location, score_government, vac_lvl))
# строим сам график
ggplot(data = data_plot_2, 
       aes(x = score_government, y = vac_lvl,
           color = factor(location))) +
  geom_point(alpha = 0.3) + geom_text(label = data_plot_2$location, check_overlap = T) +
  theme(legend.position="none")+
  scale_size_area()+
  ggtitle('Link between vaccination level and government trust by country')

# чб версия
ggplot(data = data_plot_2, 
       aes(x = score_government, y = vac_lvl)) +
  geom_point(alpha = 0.3) + geom_text(label = data_plot_2$location, 
                                      check_overlap = T, size = 6) +
  theme(legend.position="none")+
  scale_size_area()+
  theme(axis.title = element_text(size=22))



# Как уровень вакцинации влияет на прирост вакцинируемых
# выбираем интересующие столбцы
data_plot_3 <- data_dif %>% dplyr::select(location, vac_lvl, dif_vac)

# строим график
ggplot(data = data_plot_3, aes(x = vac_lvl, y = dif_vac,
                               color =  factor(location)))+geom_point()+
  ggtitle('How vaccination level affects the increase of vaccinated')


# Чб версия

ggplot(data = data_plot_3, aes(x = vac_lvl, y = dif_vac))+
  geom_point(size = 2)+
  theme(axis.title = element_text(size=22))+
  geom_smooth(method = 'lm', formula = y ~ x + I(x^2))+
  ylim(0, 32)




# Построим график измениния уровня демократии
# для всех стран
data_plot_4 <- data_dif_complicated %>% dplyr::select(date, democracy_lvl, location)
ggplot(data = data_plot_4, aes(x = date, y = democracy_lvl, 
                               group = location, color = location)) + geom_line()
# для России
data_plot_5 <- subset(data_plot_4, location == 'Russia')
data_plot_5$month <- 1:13
ggplot(data = data_plot_5, aes(x = month, y = democracy_lvl)) + geom_line()
# для Индии
data_plot_6 <- subset(data_plot_4, location == 'India')
data_plot_6$month <- 1:13
ggplot(data = data_plot_6, aes(x = month, y = democracy_lvl)) + geom_line()




# экспорт таблиц ----------------------------------------------------------
# Модели из определения спецификации
stargazer(mod1, mod2, mod3, mod7,
          se = list(cse(mod1),cse(mod2),cse(mod3),cse(mod7)),
          column.labels = c('Базовая модель', 'Модель без ВВП на душу населения', 
                            'Модель без выбросов', 'Модель с квадратом'),
          df = TRUE, intercept.top = TRUE, intercept.bottom = FALSE,
          digits = 3, decimal.mark = ',', digit.separator = '',
          type = 'latex', out = 'specification.tex')

# Количество наблюдений
stargazer(data_obs_num, type="html", out = 'num_obs.html')

# модели с квадратами

stargazer(reg_pool_2, reg_within_2, reg_within_time_2,  reg_RE_2,
          se = list(clse(reg_pool_2), clse(reg_within_2), 
                    clse(reg_within_time_2), clse(reg_RE_2)),
          column.labels = c('Модель Пула', 'Фиксированные эффекты', 
                            'Фиксированные эффекты с фиктивными времени', 
                            'Рандомные эффекты'),
          df = TRUE, intercept.top = TRUE, intercept.bottom = FALSE,
          digits = 3, decimal.mark = ',', digit.separator = '',
          type = 'latex', out = 'quadratic_models.tex')

# модели линейные

stargazer(reg_pool_2_1, reg_within_2_1, reg_within_time_2_1,  reg_RE_2_1,
          se = list(clse(reg_pool_2_1), clse(reg_within_2_1), 
                    clse(reg_within_time_2_1), clse(reg_RE_2_1)),
          column.labels = c('Модель Пула', 'Фиксированные эффекты', 
                            'Фиксированные эффекты с фиктивными времени', 
                            'Рандомные эффекты'),
          df = TRUE, intercept.top = TRUE, intercept.bottom = FALSE,
          digits = 3, decimal.mark = ',', digit.separator = '',
          type = 'latex', out = 'linear_models.tex')

# сравнение обычных моделей на модифицированных данных и чистых

stargazer(reg_pool_1, reg_pool_2, reg_pool_1_1,  reg_pool_2_1,
          se = list(clse(reg_pool_1), clse(reg_pool_2), 
                    clse(reg_pool_1_1), clse(reg_pool_2_2)),
          column.labels = c('Чистые данные, пул с квадратом', 
                            'Дисагрегированные данные, пул с квадратом', 
                            'Чистые данные, пул линейный', 
                            'Дисагрегированные данные, пул линейный'),
          df = TRUE, intercept.top = TRUE, intercept.bottom = FALSE,
          digits = 3, decimal.mark = ',', digit.separator = '',
          type = 'latex', out = 'ols_models.tex')

# сравнение моделей фиксированных эффектов

stargazer(reg_within_time_2, reg_within_time_2_1,
          se = list( clse(reg_within_time_2), 
                     clse(reg_within_time_2_1)),
          column.labels = c('С квадратом', 'Линейная'),
          df = TRUE, intercept.top = TRUE, intercept.bottom = FALSE,
          digits = 3, decimal.mark = ',', digit.separator = '',
          type = 'html', out = 'FE_models.html')

