#include "filterbank.h"
#include <opencv2/highgui.hpp>
#include <QFileInfo>
#include <QDir>

// Crea las matrices de filtros
FilterBank::FilterBank(int size): size(size), filters(38) {
    if(size < 3 || size%2 != 1) {
        std::cerr << "The filters size must be odd and higher than 2." << std::endl;
        return;
    }
    // 3 Escalas para el tamaño de los filtros de primera y segunda derivada
    int scale, scales[] = {1, 2, 4};
    // 6 orientaciones distintas
    int i, j, k, count = 0, norient = 6, nedge = norient*3;
    double angle;
    int mid_point = (size - 1)/2;
    cv::Mat points(2, size*size, CV_32FC1);

    // MR8 filter bank (vector of cv::Mat)
    for(i = 0; i < 38; i++)
        filters[i].create(size, size, CV_32FC1);

    for(i = -mid_point, k = 0; i <= mid_point; i++) {
        for(j = mid_point; j >= -mid_point; j--, k++) {
            points.at<float>(0, k) = i;
            points.at<float>(1, k) = j;
        }
    }

    cv::Mat reoriented_points(2, size*size, CV_32FC1);

    // Crea lo filtros segun orientacion y escala
    for(i = 0; i < 3; i++) {
        scale = scales[i];
        for(j = 0; j < norient; j++) {
            angle = (M_PI*j)/norient;
            reorient(points, angle, reoriented_points);
            setFilter(filters[count], reoriented_points, scale, 0, 1);
            setFilter(filters[count+nedge], reoriented_points, scale, 0, 2);
            count++;
        }
    }
    setGaussianFilter(filters[36], points, 10);
    setLoGFilter(filters[37], points, 10);

}

// Reordena el filtro de primera y segunda derivada de gaussiana
// en 6 posiciones y 3 tamanos (36 filtros en total)
void FilterBank::reorient(cv::Mat &points, float angle, cv::Mat &reoriented_points) {
    float s = sin(angle), c = cos(angle);
    cv::Mat rotation(2, 2, CV_32FC1);
    rotation.at<float>(0,0) = c;
    rotation.at<float>(0,1) = -s;
    rotation.at<float>(1,0) = s;
    rotation.at<float>(1,1) = c;
    reoriented_points = rotation*points;
    //std::cout << "Puntos: " << points;
    //std::cout << "Reorientados: " << reoriented_points;

}

// Setea el filtro gaussiano con la escala y orientacion dadas
void FilterBank::setFilter(cv::Mat &filter, cv::Mat &points, int scale, int phasex, int phasey) {
    cv::Mat gx(1, points.cols, CV_32FC1),
            gy(1, points.cols, CV_32FC1);
    int i, j, k, p = points.cols, dim = filter.rows, dim2 = dim*dim;
    for(i=0; i<p; i++) {
        gx.at<float>(0,i) = gauss(3*scale, 0, points.at<float>(0,i),phasex);
        gy.at<float>(0,i) = gauss(  scale, 0, points.at<float>(1,i),phasey);
    }

    float val, sum = 0, mean;
    for(i=0, k=0; i<dim; i++)
        for(j=0; j<dim; j++, k++) {
            val = gx.at<float>(0,k)*gy.at<float>(0,k);
            filter.at<float>(j,i) = val;
            sum += val;
        }

    mean = sum / dim*dim;
    filter -= mean;  //Center by mean
    sum = 0;
    for(i=0; i<dim2; i++)
        sum += fabs(filter.at<float>(i));
    filter /= sum;   //Normalize
}



// Filtro gaussiano (37)
void FilterBank::setGaussianFilter(cv::Mat &filter, cv::Mat &points, float sigma) {
    int i, j, k, dim = filter.rows;
    float x, y, den = 2*sigma*sigma;


    for(i=0, k=0; i<dim; i++)
        for(j=0; j<dim; j++, k++) {
            x = points.at<float>(0,k);
            y = points.at<float>(1,k);
            filter.at<float>(i,j) = exp(-(x*x + y*y)/den)/sqrt(M_PI*den);
        }
}

// Filtro laplaciano de la gaussiana (38)
void FilterBank::setLoGFilter(cv::Mat &filter, cv::Mat &points, float sigma) {
    int i, j, k, dim = filter.rows;
    float value, x, y, var = sigma*sigma, den = 2*var;


    for(i=0, k=0; i<dim; i++)
        for(j=0; j<dim; j++, k++) {
            x = points.at<float>(0,k);
            y = points.at<float>(1,k);
            value = x*x + y*y;
            filter.at<float>(i,j) = exp(-value/den)*(value-var)/(var*var*sqrt(M_PI*den));
        }
}

// Normaliza las respuestas de los filtros
void FilterBank::normalizeResponse(std::vector<float>& response) {
    float L2 = 0.0, norm;
    int i, rsize = response.size();
    for(i=0; i<rsize; i++)
        L2 += response[i]*response[i];
    L2 = sqrt(L2);
    norm = log(1.0 + L2/0.03)/L2;
    for(i=0; i<rsize; i++)
        response[i] *= norm;
}

// Aplica los filtros a las muestras (tamaño igual al filtro)
float FilterBank::applyFilter(cv::Mat &sample, cv::Mat &filter, int size) {
    float r = 0.0;
    int i, j;
    for(i = 0; i < size; i++)
        for(j = 0; j < size; j++)
            r += sample.at<float>(i,j)*filter.at<float>(i,j);
    return r;
}

// Calcula la respuesta a los filtros de las muestras
void FilterBank::calculateFilterResponses() {
    // Tamaño del arreglo de muestras normalizadas
    int i, j, k, nsize = norm_samples.size();
    filter_responses.resize(nsize);

    float max, res, rmax;
    for(i = 0; i < nsize; i++) {
        std::vector<float> &response = filter_responses[i];
        response.resize(8);
        // Calcula la resp Max al filtro de derivadas con misma orientacion pero diferente escala
        for(j = 0; j < 6; j++) {
            max = 0;
            for(k = j*6; k < (j+1)*6; k++) {
                res = applyFilter(norm_samples[i], norm_filters[k], size);
                if(fabs(res) >= max) {
                    max = fabs(res);
                    rmax = res;
                }
            }
            response[j] = rmax;
        }
        // Calcula la respuesta al Gaussiano y LoG
        response[6] = applyFilter(norm_samples[i], norm_filters[36], size);
        response[7] = applyFilter(norm_samples[i], norm_filters[37], size);
        // Normaliza las respuestas (vector de vectores de float)
        normalizeResponse(response);
    }
}


// Dada un arreglo de puntos, obtiene muestras centradas en el punto, de tamaño igual al filtro
// a las cuales se les aplican los filtros.
void FilterBank::prepareSamplesFromPoints(cv::Mat &image, std::deque<cv::Point2i> &points) {
    //samples.resize(points.size());
    //norm_samples.resize(points.size());

    std::deque<cv::Point2i>::iterator it, end = points.end();

    cv::Rect r;
    r.width = size; r.height = size;
    int i, size2 = size/2;

    for(it = points.begin(), i = 0; it != end; it++, i++) {
        cv::Point2i &p = *it;
        r.x = p.x - size2; r.y = p.y - size2;
        cv::Mat roi(image, r);
        cv::Mat empty(size, size, CV_32FC1);
        //samples[i] = roi.clone();
        samples.push_back(roi);
        norm_samples.push_back(empty);
        normalizeBGRtoGray(samples.back(), norm_samples.back(), size);
    }
}

void FilterBank::normalizeBGRtoGray(cv::Mat &in, cv::Mat &out, int size) {
    //out.create(size, size, CV_32FC1);

    float mean = 0, sd = 0, aux;
    int i, j, index, step = in.step;
    uchar *idata = in.data;
    for(i=0; i<size; i++) {
        index = i*step;
        for(j=0; j<size; j++, index++) {
            out.at<float>(i,j) = idata[index];
            mean += idata[index];
        }
    }

    mean /= size*size;

    for(i=0; i<size; i++) {
        index = i*step;
        for(j=0; j<size; j++, index++) {
            aux = idata[index] - mean;
            sd += aux*aux;
        }
    }

    sd = sqrt(sd);
    sd /= size*size;

    //Normalization :  (x - mean) / sd
    for(i=0; i<size; i++)
        for(j=0; j<size; j++) {
            out.at<float>(i,j) -= mean;
            out.at<float>(i,j) /= sd;
        }
}

void FilterBank::normalizeFilter(cv::Mat &filter, cv::Mat &norm_filter, int size) {
    norm_filter = filter.clone();

    double norm = 0;
    int i, j;

    for(i=0; i<size; i++)
        for(j=0; j<size; j++)
            norm += fabs(filter.at<float>(i,j));

    for(i=0; i<size; i++)
        for(j=0; j<size; j++)
            norm_filter.at<float>(i,j) /= norm;
}


void FilterBank::normalizeFilters() {
    int i, fsize = filters.size();
    norm_filters.resize(fsize);

    for(i = 0; i<fsize; i++)
        normalizeFilter(filters[i], norm_filters[i], size);
}


void FilterBank::prepareSamplesCrop(std::string dir_name) {
    QFileInfoList list;
    bool first = true, done = false;
    QString image_name, image_filename;
    cv::Mat image;
    int i = -1;

    do {
        if(first) {
            QDir dir;
            dir.setPath(QString(dir_name.c_str()));
            QStringList filters;
            filters << "*.jpg" << "*.png";
            dir.setNameFilters(filters);
            dir.setFilter(QDir::Files | QDir::NoSymLinks);
            //dir.setSorting(QDir::Size | QDir::Reversed);
            dir.setSorting(QDir::Name);
            list = dir.entryInfoList();
            first = false;
            if(list.size() == 0)
                break;
            samples.resize(list.size());
            norm_samples.resize(list.size());
        }
        i++;
        if(i == list.size() - 1)
            done = true;
        QFileInfo fileInfo = list.at(i);

        image_name= fileInfo.absoluteFilePath();
        image_filename = fileInfo.fileName();
        image = cv::imread(image_name.toStdString(), 1 );
        std::cout << "Preparing image: " << image_filename.toStdString() << std::endl;
        if(image.empty()) {
            std::cerr << "PLAGUES: Error reading image " << image_name.toStdString() << std::endl;
            continue;
        }

        if(image.rows > size && image.cols > size) {
            int rh = (image.rows - size) / 2,
                rw = (image.cols - size) / 2,
                w = image.cols % 2 == 0 ? size+1 : size,
                h = image.rows % 2 == 0 ? size+1 : size;
            cv::Rect r;
            r.x = rw; r.width = w; r.y = rh; r.height = h;
            cv::Mat roi(image, r);
            roi.copyTo(samples[i]);
        } else if(image.rows > size ) {
            int rh = (image.rows - size) / 2,
                h = image.rows % 2 == 0 ? size+1 : size;
            cv::Rect r;
            r.x = 0; r.width = image.cols; r.y = rh; r.height = h;
            cv::Mat roi(image, r);
            roi.copyTo(samples[i]);
        } else if(image.cols > size) {
            int rw = (image.cols - size) / 2,
                w = image.cols % 2 == 0 ? size+1 : size;
            cv::Rect r;
            r.x = rw; r.width = w; r.y = 0; r.height = image.rows;
            cv::Mat roi(image, r);
            roi.copyTo(samples[i]);
        }

        if(samples[i].rows > size || samples[i].cols > size )
            cv::resize(samples[i], samples[i], cv::Size(size, size), 0, 0, cv::INTER_CUBIC);


        samples[i].create(size, size, CV_32FC1);

        //Make function for converting from colour image to grayscale float [-1, 1]
        cv::cvtColor(image, norm_samples[i], cv::COLOR_BGR2GRAY);




    } while(!done);


}





// filtro gaussiano
float FilterBank::gauss(float sigma, float mean, float value, int ord) {
    value = value - mean;
    float g, num = value*value, var = sigma*sigma, den = 2*var;
    g = exp(-num/den)/sqrt(M_PI*den);
    if(ord == 1)
        g = (-g)*value/var;
    else if(ord == 2)
        g = g*(num - var)/(var*var);
    return g;
}

// mostrar filtros en imagen
cv::Mat FilterBank::filterToShow(cv::Mat &filter) {
    cv::Mat toShow(filter.rows, filter.cols, CV_8UC1);

    //float sum = 0;
    double min, max;
    uchar *data = toShow.data;
    cv::minMaxLoc(filter, &min, &max, NULL, NULL);
    float diff = max - min, val;
    int i, j, dim = filter.rows;
    for(i=0; i<dim; i++)
        for(j=0; j<dim; j++) {
            val = 255.0*(filter.at<float>(i,j) - min)/diff;
            data[i*dim + j] = val > 255 ? 255 : (val < 0 ? 0 : (uchar)rint(val));
        }
    return toShow;
}

