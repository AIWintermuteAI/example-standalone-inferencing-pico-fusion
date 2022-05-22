#include "ei_run_classifier.h"

#include "hardware/adc.h"
#include <hardware/gpio.h>
#include <hardware/uart.h>
#include <pico/stdio_usb.h>
#include <stdio.h>

#include "dht11.h"

const uint LED_PIN = 25;
DHT11 dht(18);

// Allocate a buffer here for the values
float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { };

bool init_sensors() {
    adc_init();
    adc_gpio_init(26);
    adc_select_input(0);

    if (!dht.begin()) {
        return false;
    }
    return true;

}

int raw_feature_get_data()
{
    for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3) {
        // Determine the next tick (and then sleep later)
        uint64_t next_tick = ei_read_timer_us() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

        dht.read();        
        features[ix + 0] = dht.readTemperature();
        features[ix + 1] = dht.readHumidity();
        features[ix + 2] = adc_read();

        sleep_us(next_tick - ei_read_timer_us());
    }

  return 0;
}

int main()
{
  stdio_usb_init();

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

    if (!init_sensors()) {
        while(1) {
        ei_printf("Sensor init error!\n");
        ei_sleep(1000);
        }
    }

  ei_impulse_result_t result = {nullptr};

  while (true)
  {
    ei_printf("Edge Impulse standalone inferencing (Raspberry Pi Pico)\n");

    while (1)
    {
      // blink LED
      gpio_put(LED_PIN, !gpio_get(LED_PIN));
      raw_feature_get_data();

      // Turn the raw buffer in a signal which we can the classify
      signal_t signal;
      int err = numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
      if (err != 0) {
          ei_printf("Failed to create signal from buffer (%d)\n", err);
          return true;
      }

      // Run the classifier
      ei_impulse_result_t result = { 0 };

      err = run_classifier(&signal, &result, false);
      if (err != EI_IMPULSE_OK) {
          ei_printf("ERR: Failed to run classifier (%d)\n", err);
          return true;
      }

      ei_printf("run_classifier returned: %d\n", err);

      if (err != 0)
        return true;

      ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);

      // print the predictions
      ei_printf("[");
      for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
      {
        ei_printf("%.5f", result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ei_printf(", ");
#else
        if (ix != EI_CLASSIFIER_LABEL_COUNT - 1)
        {
          ei_printf(", ");
        }
#endif
      }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
      printf("%.3f", result.anomaly);
#endif
      printf("]\n");

      ei_sleep(2000);
    }
  }
}