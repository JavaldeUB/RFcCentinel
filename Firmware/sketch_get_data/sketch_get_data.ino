/* Requirements:
 *   - CPU Speed must be either 120 or 240 Mhz. Selected via "Menu -> CPU Speed"
 *   - USB Stack must be Adafruit Tiny USB
 */

#include "net_definitions.h"
#include "usbh_helper.h"
Adafruit_USBH_CDC SerialHost;

#define MAX_VALUES 201           // array maximum size, from 0 to 100 (101 values)
#define MAXIMUM_dB -0.21

float dB_values[MAX_VALUES];     // array to store measurements magnitudes
int dB_count = 0;                // array elements counter

    float res_vec[3];
  float w_vec[30];
  float bias_vec[30];
  float H_vec[30];


float logsig(float x) {
    return 1.0 / (1.0 + exp(-x));
}


void ann(){
  int i,j,ind_max=0;
  float max=-1000;


  for(j=0;j<30;j++)
  {
    w_vec[j]=0;
    bias_vec[j]=0;
    for(i=0;i<201;i++)
    {
      w_vec[j]=w_vec[j]+dB_values[i]*w[i][j];
    }
    bias_vec[j]=w_vec[j]+phi[j];
    H_vec[j]=logsig(bias_vec[j]);
  }

  for(j=0;j<3;j++)
  {
    res_vec[j]=0;
    for(i=0;i<30;i++)
    {
      res_vec[j]=res_vec[j]+H_vec[i]*B[i][j];
    }
  }


}

// forward Serial <-> SerialHost
void forward_serial(void) {
uint8_t buf[128];         // data is received from nano VNA as unsigned char
char dataStr[128];        // data stored as string
float real, imag, magnitude;
  int i,j,ind_max=0;
  float max=-1000;


  // From PC to Nano VNA
  if (Serial.available()) {
    size_t count = Serial.read(buf, sizeof(buf));
    if (SerialHost && SerialHost.connected()) {
      SerialHost.write(buf, count);
      SerialHost.flush();
    }
  }

  // From Nano VNA to PC
  if (SerialHost.connected() && SerialHost.available()) {
    size_t count = SerialHost.read(buf, sizeof(buf));         // read data received from nano VNA
    strncpy(dataStr, (char *)buf, count);                     // Change data from nano VNA to text to obtain real and imaginary parts
    dataStr[count] = '\0';                                    // End of string with null command

    if (sscanf(dataStr, "%f %f", &real, &imag) == 2) {        // split data in real and imaginary parts and compute the magnitude
      magnitude = sqrt(real * real + imag * imag);
      float dB = 20 * log10(magnitude);
      Serial.print("Magnitude (dB): ");
      Serial.println(dB, 3);

      if (dB_count < MAX_VALUES) {                            // Save dB measurement to array
        dB_values[dB_count] = dB;
        dB_count++;
      }
      
      if (dB_count == MAX_VALUES) {                           // when all measurements received compute mean value
        float sum = 0;
        for (int i = 0; i < MAX_VALUES; i++) {
          sum += dB_values[i];
        }
        float average_dB = sum / MAX_VALUES;
        Serial.println();
        Serial.print("Received a total of ");
        Serial.print(dB_count);
        Serial.println(" measurements.");
        Serial.print("Average Magnitude (dB): ");
        for (i=0)
        Serial.println(average_dB, 6);
        //for (int i = 0; i < MAX_VALUES; i++) {
        //  dB_values[i] = dB_values[i]/maximum_db[i];
        //}
        ann();
        //Maximum calculation and Type decodification
        Serial.println("");
        Serial.println("**NEW SAMPLE**");
        Serial.println("***RESULTS***");
        for(i=0;i<3;i++)
        {
          if (res_vec[i]>max)
          {
            ind_max=i;
            max=res_vec[i];
          }
          Serial.print(res_vec[i],3);
          Serial.print(" ");
        }
        switch(ind_max)
        {
          case 0:
          Serial.println("Type: Arbequina");
          break;
          case 1:
          Serial.println("Type: Cornicabra");
          break;
          case 2:
          Serial.println("Type: Picual");
          break;
          default:
          Serial.println("Unknown");
        }
        dB_count = 0;                                         // Restart counter for next measurement
      }
    }
  }
}

//------------- Core0 -------------//
void setup() {
  Serial.begin(115200);
  Serial.println("Nano VNA connection");
}

void loop() {
  forward_serial();
}

//------------- Core1 -------------//
void setup1() {
  // configure pio-usb: defined in usbh_helper.h
  rp2040_configure_pio_usb();
  USBHost.begin(1);
  SerialHost.begin(115200);
}

void loop1() {
  USBHost.task();
}

//--------------------------------------------------------------------+
// TinyUSB Host callbacks
//--------------------------------------------------------------------+
extern "C" {

// Invoked when a device with CDC interface is mounted
// idx is index of cdc interface in the internal pool.
void tuh_cdc_mount_cb(uint8_t idx) {
  // bind SerialHost object to this interface index
  SerialHost.mount(idx);
  Serial.println("SerialHost is connected to a new CDC device");
}

// Invoked when a device with CDC interface is unmounted
void tuh_cdc_umount_cb(uint8_t idx) {
  SerialHost.umount(idx);
  Serial.println("SerialHost is disconnected");
}

}