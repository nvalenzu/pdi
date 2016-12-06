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
    // Consigue todas las muestras posibles de las imagenes capturadas
    for (int i = 0; i < (int)(ImageToPoints.cols/filterSize); i++) {
        for (int j = 0; j < (int)(ImageToPoints.rows/filterSize); j++) {
            cv::Point2i p((filterSize-1)/2 + i*filterSize, (filterSize-1)/2 + j*filterSize);
            points.push_back(p);
        }
    }
    // Toma muestra solo con punto central (si se usa, comentar anterior)
    //cv::Point2i p(ImageToPoints.cols/2, ImageToPoints.rows/2);
    //points.push_back(p);
}

int main(int argc, char *argv[])
{
    QFileInfo file(argv[1]);
    QFileInfoList list;
    QDir dir;
    int bankSize = 49;

    // Carga el directorio inicial
    if (argc == 1) {
         std::cerr << "Error: please add directory path" << std::endl;
         return -1;
    }

    if(!file.exists()) {
        std::cerr << "Unable to open file or directory '" << argv[1];
        std::cerr << "'. It does not exist." << std::endl;
        return -1;
    }

    // Guarda en una lista los directorios con las distintas texturas
    std::string rootDir = argv[1];
    QDir rootD;
    QFileInfoList dirList;
    rootD.setPath(QString(rootDir.c_str()));
    dirList = rootD.entryInfoList();

    // Crear un banco de filtros para cada textura
    std::vector<FilterBank> fbank;

    // Crea una clase contenedora para cada tipo de muestra
    for (int n = 0; n < dirList.size()-2; n++) {
        //1. Create Filter Bank
        fbank.emplace_back(bankSize);

        //2. Normalize filters by L1
        fbank[n].normalizeFilters();

        //3. Obtain samples and normalized samples from image points: Leave samples equal to the filter size
        std::deque<cv::Point2i> imagePoints;
        // n+2, n = 0 directorio "." y n = 1 directorio ".."
        if(dirList.at(n+2).isDir()) {
            // Genera una lista con las imagenes en el directorio
            std::string name = dirList.at(n+2).absoluteFilePath().toStdString();
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
                fbank[n].prepareSamplesFromPoints(image, imagePoints);
                // Mostrar imagenes
                //QString s = "Imagen " + QString::number(n) + QString::number(i);
                //cv::imshow(s.toStdString(), image);
            }
            // Agrega nombre al banco
            QString nombre = QString::fromStdString(name);
            nombre.replace(QString(argv[1]), QString(""));
            fbank[n].name = nombre.toStdString();
        }


//        //Show samples
//        for(unsigned i = 0; i < fbank[n].samples.size(); i++) {
//            QString s = "Bank: Sample " + QString::number(i);
//            cv::Mat r;
//            cv::resize(fbank[n].samples[i],r, cv::Size(2*bankSize, 2*bankSize));
//            cv::imshow(s.toStdString(), r);
//        }
        cv::waitKey(0);

        //4. Obtain filter responses
        fbank[n].calculateFilterResponses();

        //5. Aplicar K-means
        fbank[n].applyKmeans(fbank[n].filter_responses.size());

        //std::cout << "Numero de muestras analizadas: " << fbank[n].filter_responses.size() << std::endl;
        std::cout << "Kpoints: " << std::endl;
        std::cout << fbank[n].Kpoints << std::endl;
        std::cout << "TextonDictionary: " << std::endl;
        std::cout << fbank[n].TextonDictionary << std::endl;
        //std::cout << "BestLabels: " << std::endl;
        //std::cout << fbank[n].bestLabels << std::endl;
        std::cout << std::endl;
    }

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
    //cv::Mat r;
    //cv::resize(trainingBank.samples[0],r, cv::Size(2*bankSize, 2*bankSize));
    //cv::imshow("Imagen entrenamiento", r);
    // Calcular respuesta a filtros de imagen de entrenamiento
    trainingBank.calculateFilterResponses();

    // Transformar respuesta de tipo "vector" a "Mat"
    std::vector<float> temp = trainingBank.filter_responses[0];
    cv::Mat Tresponse(1, temp.size(), CV_32F);
    for (unsigned i = 0; i < temp.size(); i++)
        Tresponse.at<float>(0, i) = temp.at(i);

    // Mostrar respuesta a filtros imagen de entrenamiento
    std::cout << "Respuesta imagen entrenamiento: " << std::endl;
    std::cout <<  Tresponse << std::endl;
    std::cout << std::endl;

    // Buscar distancia minima entre img de entrenamiento y diccionario de textones
    for (unsigned i = 0; i < fbank.size(); i++){
        float min = 9999;
        float dist;
        int c = 0;
        for (int j = 0; j < fbank[i].TextonDictionary.rows; j++) {
            dist = cv::norm(fbank[i].TextonDictionary.row(j), Tresponse);
            if (dist < min) {
                min = dist;
                c = j;
            }
        }
        std::cout << fbank[i].name << " - ";
        std::cout <<  "Distancia minima encontrada en el texton " << c << " es: " << min << std::endl;
    }

    // Calcular distancia minima para solo un filtro (no vector completo) con cada texton, y calcular
    // la cantidad de textones que se agrupan a ese texton (antes del K-means). INCOMPLETO
    //float min = 999;
    //int x;
    //for (int i = 0; i < fbank[0].TextonDictionary.rows; i++) {
    //  float dist;
    //  float d1 = fbank[0].TextonDictionary.at<float>(i,0);
    //  float d2 = Tresponse.at<float>(0,0);
    //  dist = cv::abs(d1 - d2);
    //  if(dist < min) {
    //      min = dist;
    //      x = i;
    //  }
    //  cout << d1 << " - "<< d2 << " = " << dist << endl;
    //}
    //cout << min << "en: " << x << endl;

    // Histogramas ???
    //int hist[10];
    //std::fill_n(hist, 10, 0);
    //for (int i = 0; i < bestLabels.rows; i++) {
    //    hist[bestLabels.at<int>(i)]++;
    //}
    //for (int i = 0; i < 10; i++)
    //    cout << "hay " << hist[i] <<  " " << i << "s" << endl;

    // Show filters
    //for(int i = 0; i < 38; i++) {
        //QString s = "Bank: Filter " + QString::number(i);
        //cv::Mat r;
        //cv::resize(FilterBank::filterToShow(fbank[0].filters[i]),r, cv::Size(2*bankSize, 2*bankSize));
        //cv::imshow(s.toStdString(), r);
    //}
    //cv::waitKey(0);
    return 0;
}
