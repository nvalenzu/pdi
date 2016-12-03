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
    std::deque<cv::Point2i> imagePoints;

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
        cv::resize(FilterBank::filterToShow(fbank.filters[i]),r, cv::Size(4*bankSize, 4*bankSize));
        cv::imshow(s.toStdString(), r);
    }*/

    //2. Normalize filters by L1
    fbank.normalizeFilters();

    //3. Obtain samples and normalized samples from image points: Leave samples equal to the filter size
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
        cv::resize(fbank.samples[i],r, cv::Size(4*bankSize, 4*bankSize));
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
    cv::Mat bestLabels, centers;
    cv::Mat Kpoints(fbank.filter_responses.size(), 8, CV_32F);
    std::cout << "Numero de muestras analizadas: " << fbank.filter_responses.size() << std::endl;

    for(unsigned j = 0; j < fbank.filter_responses.size(); j++) {
        std::vector<float> asd = fbank.filter_responses[j];
        for (unsigned i = 0; i < asd.size(); i++) {
            Kpoints.at<float>(j, i) = asd.at(i);
            std::cout << asd.at(i) << " ";
        }
        std::cout << std::endl;
    }
    cv::kmeans(Kpoints, 6, bestLabels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    std::cout << "Kpoints: " << std::endl;
    std::cout << Kpoints << std::endl;
    std::cout << "CENTROS: " << std::endl;
    std::cout << centers << std::endl;
    std::cout << "BestLabels: " << std::endl;
    std::cout << bestLabels << std::endl;




    return 0;
}
