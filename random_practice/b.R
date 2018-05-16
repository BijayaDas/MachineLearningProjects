mydata <- read.csv(file.choose())
install.packages("ggplot2")

ggplot(data=mydata, aes(x=carat, y=price, color=clarity)) + geom_point(alpha=0.1) + geom_smooth()

