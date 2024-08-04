---
id: 698
title: 'Coding with Karthik: Perceptron Learning Algorithm in C'
date: '2024-05-25T21:10:23+00:00'

layout: post
guid: 'https://www.karthikvedula.com/?p=698'
permalink: /2024/05/25/coding-with-karthik-perceptron-learning-algorithm-in-c/
categories:
    - 'Coding with Karthik'
    
tags:
    - 'C Programming Language'
    - 'Perceptron Learning'
---

[Coding with Karthik: Visualizing the Perceptron Learning Algorithm]({{ site.baseurl }}/2024/01/05/visualizing-the-perceptron-learning-algorithm/)

Remember that blog post? ^

Well, I wanted to do the same thing but in the **C Programming Language**.

<img src="{{ site.baseurl }}/assets/images/2024/05/IMG_1990-1-771x1024.jpg" alt="K&R Book" width="200"/>

To write the perceptron learning algorithm in C, I had to let go of my comforts. File reading became a task of its own. I spent a couple hours trying to figure out pointers. Hopefully I don’t have memory leaks – I think `free()`‘d whenever I `malloc()`‘d. 😀

Below is the code. Enjoy!
```c
/*
 * LOGIC:
 * 1. Start with random B, W1, W2,
 * 2. get_misclassified(B, W1, W2)
 * 3. for sample in misclassified:
 *        get values x1, x2, y (y is binary)
 *        if y == 0: y = -1
 *        B = B + y * 1 * LR
 *        W1 = W1 + y * x1 * LR
 *        W2 = W2 + y * x2 * LR
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE_LENGTH 100

typedef struct datapoint {
  double x1;
  double x2;
  bool y;
} datapoint;

const double LR = 1; // learning rate
const int STOP_ITER = 10000; // maximum number of epochs run
int dataset_size;

// generate random decimal between -1 and 1
double gen_random_neg1_1() { 
  double r = -1 + 2 * ((float)rand()) / RAND_MAX;
  return r;
}

bool gen_true_false() {
  int i = rand() % 2;

  if (i == 1) {
    return true;
  } else {
    return false;
  }
}

datapoint *get_misclassified(datapoint *dataset, int dataset_len, double b, double w1, double w2, int *pmisclassified_count) {

  datapoint *misclassified_points;
  misclassified_points =
      malloc(dataset_len * sizeof(datapoint)); // maximum the list can go

  *pmisclassified_count = 0;

  for (int i = 0; i < dataset_len; i++) {
    datapoint sample = dataset[i];
    double pred = b + w1 * sample.x1 + w2 * sample.x2;

    if (pred > 0 != sample.y) {
      misclassified_points[*pmisclassified_count] = sample;
      (*pmisclassified_count)++;
    }
  }

  return misclassified_points;
}

int get_file_len(char *filename) {
  FILE *pfile = fopen(filename, "r");
  if (pfile == NULL) {
    printf("No file %s", filename);
    exit(1);
  }

  int lines = 0;

  while (!feof(pfile)) {
    int ch = fgetc(pfile);
    if (ch == '\n') {
      lines++;
    }
  }

  return lines;
}

datapoint *get_dataset(char *filename) {
  FILE *pfile = fopen(filename, "r");
  if (pfile == NULL) {
    printf("No file %s", filename);
    return 0;
  }

  dataset_size = get_file_len(filename);

  int count = 0;

  datapoint *dataset;
  dataset = malloc(dataset_size * sizeof(datapoint));

  char line[MAX_LINE_LENGTH];

  while (fgets(line, MAX_LINE_LENGTH, pfile) != NULL) {
    line[strcspn(line, "\n")] = '\0'; // strip newline
    char *token = strtok(line, ",");
    if (token != NULL) {
      dataset[count].x1 = atof(token);
      token = strtok(NULL, ",");
      if (token != NULL) {
        dataset[count].x2 = atof(token);
        token = strtok(NULL, ",");
        if (token != NULL) {
          dataset[count].y = atoi(token);
          count++;
        }
      }
    }
  }

  fclose(pfile);

  return dataset;
}

int main(int argc, char *argv[]) {
  srand(time(NULL)); // Initialization, should only be called once.

  double b = gen_random_neg1_1();
  double w1 = gen_random_neg1_1();
  double w2 = gen_random_neg1_1();

  datapoint *dataset = get_dataset(argv[1]);

  // get array of misclassified points and the length of the array
  int misclassified_count;
  datapoint *misclassified =
      get_misclassified(dataset, dataset_size, b, w1, w2, &misclassified_count);

  int count = 0; // count number of epochs

  while (misclassified_count > 0) {
    for (int i = 0; i < misclassified_count; i++) {
      // get sample and update weights

      datapoint sample = misclassified[i];
      int multiplier = sample.y ? 1 : -1;

      b = b + multiplier * LR;
      w1 = w1 + multiplier * sample.x1 * LR;
      w2 = w2 + multiplier * sample.x2 * LR;
    }

    free(misclassified);
    misclassified = get_misclassified(dataset, dataset_size, b, w1, w2, &misclassified_count);

    count++;

    if (count > STOP_ITER) {
      break;
    }
  }

  free(dataset);
  free(misclassified);

  printf("FINAL VALUES\n");
  printf("b:  %f\n", b);
  printf("w1: %f\n", w1);
  printf("w2: %f\n", w2);
  printf("Misclassified count %d", misclassified_count);
  printf("\n\n");

  return 0;
}
```
