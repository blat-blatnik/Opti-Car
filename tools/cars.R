library(tidyverse)
library(ggplot2)
library(MASS)
library(gridExtra)

dat <- read.csv(choose.files())

# for some reason R converts some of these to factors even though they are clearly numbers
# so we need to convert them back here...
dat$MSRP <- as.numeric(dat$MSRP)
dat$Year <- as.numeric(dat$Year)
dat$FrontWheelSize <- as.numeric(dat$FrontWheelSize)
dat$Displacement <- as.numeric(dat$Displacement)
dat$GasMileage <- as.numeric(dat$GasMileage)
dat$Width <- as.numeric(dat$Width)
dat$Height <- as.numeric(dat$Height)
dat$Length <- as.numeric(dat$Length)
dat$NumDoors <- as.numeric(dat$NumDoors)

model <- lm(MSRP ~ Make + Year + Horsepower + EngineType + Width + Height + Length + BodyStyle + NumDoors, data=dat)
model <- stepAIC(model, trace=3)
summary(model)

mean.price = mean(dat$MSRP, na.rm=TRUE)
p1 <- ggplot(dat, aes(log10(MSRP))) +
	geom_histogram(bins=30) +
	geom_vline(xintercept=log10(mean.price), color='red', linetype='dashed', size=1.25) +
	theme_classic(base_size=18) +
	scale_x_continuous(breaks=c(4.0, log10(30000), 5.0, log10(300000)), labels=c('10,000', '30,000', '100,000', '300,000')) +
	labs(title='Car price distribution', 
		 x='price ($)', 
		 y='count')

years <- count(dat, Year)
p2 <- ggplot(years, aes(x=Year, y=n)) +
	geom_bar(stat='identity', position='dodge', color='black', width=0.7) +
	theme_classic(base_size=18) +
	theme(axis.text.y=element_text(face='italic', size=rel(0.85)), legend.position='none') +
	labs(title='Car manufacture year distribution', 
		 x='year', 
		 y='count')

makes <- count(dat, Make)
p3 <- ggplot(makes, aes(x=reorder(Make, n), y=n)) +
	geom_bar(stat='identity', position='dodge', color='black', width=0.7) +
	coord_flip() +
	theme_classic(base_size=18) +
	theme(axis.text.y=element_text(face='italic', size=rel(0.4)), legend.position='none') +
	labs(title='Car make distribution', 
		 x='make', 
		 y='count')

bodies <- count(dat, BodyStyle)
levels(bodies$BodyStyle)[1] <- 'not specified'
levels(bodies$BodyStyle)[2] <- '2 door'
levels(bodies$BodyStyle)[3] <- '3 door'
levels(bodies$BodyStyle)[4] <- '4 door'
p4 <- ggplot(bodies, aes(x=reorder(BodyStyle, n), y=n)) +
	geom_bar(stat='identity', position='dodge', color='black', width=0.7) +
	coord_flip() +
	theme_classic(base_size=18) +
	theme(axis.text.y=element_text(face='italic', size=rel(1.0)), legend.position='none') +
	labs(title='Car body style distribution', 
		 x='body style', 
		 y='count')

grid.arrange(p1, p2, p3, p4)
