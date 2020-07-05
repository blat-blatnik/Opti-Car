from glob import glob
from sys import argv
from os import path

if len(argv) == 1:
	print('usage: $ %s PATTERN' % argv[0])
	print('example: $ %s "Database/*.jpg"' % argv[0])
	exit()

with open('cars.csv', 'w') as cars:
	cars.write('ID,MSRP,Make,Model,Year,FrontWheelSize,Horsepower,Displacement,EngineType,Width,Height,Length,GasMileage,Drivetrain,PassangerCapacity,NumDoors,BodyStyle\n')
	pattern = argv[1]
	for file in glob(pattern):
		file = path.basename(file)
		print(file)
		parts = file.replace('.jpg', '').replace('nan', '')
		Make,Model,Year,MSRP,FrontWheelSize,Horsepower,Displacement,EngineType,Width,Height,Length,GasMileage,Drivetrain,PassangerCapacity,NumDoors,BodyStyle,ID = parts.split('_')
		try:
			newMSRP = str(int(MSRP) * 1000)
			MSRP = newMSRP
		except:
			toobad=0
		dat = ','.join([ID,MSRP,Make,Model,Year,FrontWheelSize,Horsepower,Displacement,EngineType,Width,Height,Length,GasMileage,Drivetrain,PassangerCapacity,NumDoors,BodyStyle])
		cars.write(dat + '\n')
	cars.write('\n')