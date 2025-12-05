#include <ESP32Servo.h>
#include <math.h>
Servo servoX;
Servo servoY;
int servoPinX = 6;
int servoPinY = 5;
int currentAngle_X = 90;
int currentAngle_Y = 90;
double Kp = 0.03;
double Ki = 0.0004;
double Kd = 0.5;
double dt = 0;
double previousTime = 0;
double integral = 0;
double previousError = 0;
void setup() {
  // put your setup code here, to run once:

 Serial.begin(115200);
 Serial.println("ESP32 Ready to Receive");
 servoX.attach(servoPinX);
 servoY.attach(servoPinY);
 servoX.write(currentAngle_X);
 servoY.write(currentAngle_Y);
  
}


void loop() {
  // put your main code here, to run repeatedly:
  
   if(Serial.available()){
    //servoX.write(65);
    //servoY.write(90);
    double currentTime = millis();
    dt = (currentTime - previousTime) / 1000;
    String receivedString = Serial.readStringUntil('\n');
    String receivedString2 = Serial.readStringUntil('\n');
    Serial.print("received integers");
    int receivedInt = receivedString.toInt();
    int receivedInt2 = receivedString2.toInt();
    Serial.println("Received Integers: ");
    Serial.println(receivedInt);
    Serial.println(receivedInt2);
    int x_error = receivedInt - 319;
    int y_error = receivedInt2 - 239;
    int X_Angle = currentAngle_X - pid(x_error);
    int Y_Angle = currentAngle_Y - pid(y_error);
    servoX.write(X_Angle);
    servoY.write(Y_Angle);
    
    }

    
}

int pid(int error){
  int proportional = error;
  integral = integral + (error * dt);
  double derivative = (error - previousError) / dt;
  //double output = (Kp * proportional) * (Ki * integral) + (Kd * derivative);
  //int myOutput = (int)ceil(output);
  previousError = error;
  double output = Kp * error + Ki * integral + Kd * derivative;
  return output;
  }

  
