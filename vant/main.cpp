#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <rgb_hist.h>
#include <QImage>
#include "filterbank.h"
#include <QDir>

using namespace std;
using namespace cv;


void getPointsFromImage(std::deque<cv::Point2i> &points, cv::Mat &ImageToPoints, int filterSize) {
    points.clear();
    for (int i = 0; i < (int)(ImageToPoints.cols/filterSize); i++) {
        for (int j = 0; j < (int)(ImageToPoints.rows/filterSize); j++) {
            cv::Point2i p((filterSize-1)/2 + i*filterSize, (filterSize-1)/2 + j*filterSize);
            points.push_back(p);
        }
    }

    //cv::Point2i p(ImageToPoints.cols/2, ImageToPoints.rows/2);
    //points.push_back(p);
}

int main(int argc, char *argv[])
{
    QFileInfo file(argv[1]);
    QFileInfoList list;
    QDir dir;
    int bankSize = 49;


    if (argc == 1) {
         std::cerr << "Error: please add directory path" << std::endl;
         return -1;
    }

    if(!file.exists()) {
        std::cerr << "Unable to open file or directory '" << argv[1];
        std::cerr << "'. It does not exist." << std::endl;
        return -1;
    }

    // 1.Create filter bank
    FilterBank fbank(bankSize);

    // show filters
    /*
    for(int i = 0; i < 38; i++) {
        QString s = "Bank: Filter " + QString::number(i);
        cv::Mat r;
        cv::resize(FilterBank::filterToShow(fbank.filters[i]),r, cv::Size(2*bankSize, 2*bankSize));
        cv::imshow(s.toStdString(), r);
    }*/

    //2. Normalize filters by L1
    fbank.normalizeFilters();

    //3. Obtain samples and normalized samples from image points: Leave samples equal to the filter size
    std::deque<cv::Point2i> imagePoints;
    if(file.isDir()) {
        // Genera una lista con las imagenes en el directorio
        std::string name = file.absoluteFilePath().toStdString();
        dir.setPath(QString(name.c_str()));
        QStringList filters;
        filters << "*.jpg" << "*.png";
        dir.setNameFilters(filters);
        list = dir.entryInfoList();
        // Para cada imagen, toma las muestras
        for (int i = 0; i < list.size(); i++) {
            cv::Mat image = cv::imread(list.at(i).absoluteFilePath().toStdString());
            if(image.empty()) {
                std::cerr << "Error reading image " << list.at(i).absoluteFilePath().toStdString() << std::endl;
                return -1;
            }
            // Generar arreglo de puntos para forma cuadrado alrededor del punto
            // con tamaño igual al filtro
            getPointsFromImage(imagePoints, image, bankSize);
            fbank.prepareSamplesFromPoints(image, imagePoints);
            QString s = "Imagen " + QString::number(i);
            cv::imshow(s.toStdString(), image);
        }
    }
    //Show samples
    /*
    for(unsigned i = 0; i < fbank.samples.size(); i++) {
        QString s = "Bank: Sample " + QString::number(i);
        cv::Mat r;
        cv::resize(fbank.samples[i],r, cv::Size(2*bankSize, 2*bankSize));
        cv::imshow(s.toStdString(), r);
    }*/
    cv::waitKey(0);


    //4. Obtain filter responses
    fbank.calculateFilterResponses();

    /*
    // Cargar imagen
    if (argc == 1) {
         std::cerr << "Error abriendo imagen" << std::endl;
         return -1;
    }
    cv::Mat image = cv::imread(argv[1], 1);
    if(image.empty()) {
        std::cerr << "Error reading image " << argv[1] << std::endl;
        return 1;
    }
    cv::imshow("imagen", image);

    // Imprime puntos
    std::cout << "Arreglo de puntos: ";
    for (unsigned i = 0; i < imagePoints.size(); i++) {
        cout << imagePoints[i] <<  " ";
    }
    */

    //5. Aplicar K-means
    cv::Mat bestLabels, TextonDictionary;
    cv::Mat Kpoints(fbank.filter_responses.size(), 8, CV_32F);

    // Transformar respuestas de tipo std::vector<std::vector>> a cv::Mat Kpoints
    for(unsigned j = 0; j < fbank.filter_responses.size(); j++) {
        std::vector<float> temp = fbank.filter_responses[j];
        for (unsigned i = 0; i < temp.size(); i++) {
            Kpoints.at<float>(j, i) = temp.at(i);
        }
        std::cout << std::endl;
    }
    // Generar 10 clusters por textura
    cv::kmeans(Kpoints, 10, bestLabels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, TextonDictionary);

    //std::cout << "Numero de muestras analizadas: " << fbank.filter_responses.size() << std::endl;
    //std::cout << "Kpoints: " << std::endl;
    //std::cout << Kpoints << std::endl;
    std::cout << "TextonDictionary: " << std::endl;
    std::cout << TextonDictionary << std::endl;
    //std::cout << "BestLabels: " << std::endl;
   // std::cout << bestLabels << std::endl;


    //6. Generar modelo
    // Creamos banco nuevo para imagen de entrenamiento
    FilterBank trainingBank(bankSize);
    // Normalizar filtros
    trainingBank.normalizeFilters();
    // Leer imagen de prueba
    cv::Mat trainingImage = cv::imread("../3.jpg", 1);
    if(trainingImage.empty()) {
        std::cerr << "Error reading image " << "../3.jpg" << std::endl;
        return 1;
    }
    // Sacar punto central imagen de entrenamiento
    std::deque<cv::Point2i> trainingPoints;
    cv::Point2i centralPoint(trainingImage.cols/2, trainingImage.rows/2);
    trainingPoints.push_back(centralPoint);
    // Preparar muestra de tamaño igual al filtro
    trainingBank.prepareSamplesFromPoints(trainingImage, trainingPoints);
    // Mostrar imagen de entrenamiento
        cv::Mat r;
        cv::resize(trainingBank.samples[0],r, cv::Size(2*bankSize, 2*bankSize));
        cv::imshow("Imagen entrenamiento", r);

    // Calcular respuesta a filtros de imagen de entrenamiento
    trainingBank.calculateFilterResponses();

    // Transformar respuesta de tipo "vector" a "Mat"
    std::vector<float> trainResponse = trainingBank.filter_responses[0];
    cv::Mat Tresponse(1, trainResponse.size(), CV_32F);
    for (unsigned i = 0; i < trainResponse.size(); i++)
        Tresponse.at<float>(0, i) = trainResponse.at(i);

    // Mostrar respuesta a filtros imagen de entrenamiento
    std::cout << std::endl;
    std::cout << "Respuesta imagen entrenamiento " << std::endl;
    std::cout <<  Tresponse << std::endl;

    // Buscar distancia minima entre img de entrenamiento y diccionario
    cv::Mat distancias(TextonDictionary.rows, 1, CV_32F);
    float min = 9999;
    float dist;
    int c = 0;
    for (unsigned i = 0; i < TextonDictionary.rows; i++) {
        dist = cv::norm(TextonDictionary.row(i), Tresponse);
        if (dist < min) {
            min = dist;
            c = i;
        }
    }
    std::cout <<  "Distancia minima: " << min << " encontrada en el texton " << c << std::endl;


    // Histogramas ???
//    int hist[10];
//    std::fill_n(hist, 10, 0);

//    //cout << "sdjkhjaks: " << roi.cols*roirows << endl;

//    for (int i = 0; i < bestLabels.rows; i++) {
//        hist[bestLabels.at<int>(i)]++;
//    }

//    for (int i = 0; i < 10; i++)
//        cout << "hay " << hist[i] <<  " " << i << "s" << endl;



    waitKey(0);


    return 0;
}
