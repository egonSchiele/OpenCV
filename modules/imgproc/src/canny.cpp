/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

CV_IMPL void cvCanny( const void* srcarr, void* dstarr,
                      double low_thresh, double high_thresh,
                      int aperture_size )
{
    cv::Ptr<CvMat> dx, dy;
    cv::AutoBuffer<char> buffer;
    std::vector<uchar*> stack;
    uchar **stack_top = 0, **stack_bottom = 0;

	double percent = low_thresh;

    CvMat srcstub, *src = cvGetMat( srcarr, &srcstub );
    CvMat dststub, *dst = cvGetMat( dstarr, &dststub );
    CvSize size;
    int flags = aperture_size;
    int low, high;
    uchar* map;
    ptrdiff_t mapstep;
    int maxsize;
    int i, j;
    CvMat mag_row;

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
        CV_MAT_TYPE( dst->type ) != CV_8UC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( low_thresh > high_thresh )
    {
        double t;
        CV_SWAP( low_thresh, high_thresh, t );
    }

    aperture_size &= INT_MAX;
    if( (aperture_size & 1) == 0 || aperture_size < 3 || aperture_size > 7 )
        CV_Error( CV_StsBadFlag, "" );

    size = cvGetMatSize( src );

	// convolve with sobel operator to get derivative approximations
    dx = cvCreateMat( size.height, size.width, CV_16SC1 );
    dy = cvCreateMat( size.height, size.width, CV_16SC1 );
    cvSobel( src, dx, 1, 0, aperture_size );
    cvSobel( src, dy, 0, 1, aperture_size );

    if( flags & CV_CANNY_L2_GRADIENT )
    {
        Cv32suf ul, uh;
        ul.f = (float)low_thresh;
        uh.f = (float)high_thresh;

        low = ul.i;
        high = uh.i;
    }
    else
    {
        low = cvFloor( low_thresh );
        high = cvFloor( high_thresh );
    }


	// buffer structure will be: top half for 2d mag array,
	// bottom half for map of edges (either 0, 1, or 2...see below)
    buffer.allocate( (size.width+2)*(size.height+2) + (size.width + 2)*(size.height+2)*sizeof(int) );

	// mag is a pointer to the magnitude array
    int *mag = (int*)(char*)buffer;

	// map is a pointer to the edges array
    map = (uchar*)(mag + (size.width+2)*(size.height+2));
    
	mapstep = size.width + 2;

    maxsize = MAX( 1 << 10, size.width*size.height/10 );
    stack.resize( maxsize );
    stack_top = stack_bottom = &stack[0];

    memset( mag, 0, (size.width + 2) * (size.height + 2) * sizeof(int) );
    memset( map, 1, mapstep );
    memset( map + mapstep*(size.height + 1), 1, mapstep );

    /* sector numbers 
       (Top-Left Origin)

        1   2   3
         *  *  * 
          * * *  
        0*******0
          * * *  
         *  *  * 
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    mag_row = cvMat( 1, size.width, CV_32F );
	
	// we actually want to start from (1,1), because there's a 1-cell border
	// around the whole image for padding. 
	mag = mag + size.width + 2 + 1;

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
		// here we move one column over, b/c the first column is padding.	
		int *_mag = mag + (size.width + 2) * i;
        float* _magf = (float*)_mag;
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        int x, y;

        if( i < size.height ) {
            _mag[-1] = _mag[size.width] = 0;

            if( !(flags & CV_CANNY_L2_GRADIENT) ) {
                for( j = 0; j < size.width; j++ ) {
                    _mag[j] = abs(_dx[j]) + abs(_dy[j]);
				}
			}
            else {
                for( j = 0; j < size.width; j++ ) {
                    x = _dx[j]; y = _dy[j];
                    _magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
                }
            }
			
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(int) );
	}

	// Choose better thresholds
	int max = 0;
	for (i = 0; i < size.height; i++) {
		int *_mag = mag + (size.width + 2) * i;
		for( j = 0; j < size.width; j++ ) {
			if (_mag[j] > max) {
				max = _mag[j];
			}
		}
	}
	
	// step 2: Get the histogram of the data.
#define NUM_BINS 64
	// might want to make this max - min / NUM_BINS after you have normalized.
	int bin_size = max / NUM_BINS;
	if (bin_size < 1) bin_size = 1;
	int bins[NUM_BINS] = { 0 };

	for (i = 0; i < size.height; i++) {
		int *_mag = mag + (size.width + 2) * i;
		for( j = 0; j < size.width; j++ ) {
			bins[_mag[j] / bin_size]++;
		}
	}

	// step 3: get the high threshold
	double percent_of_pixels_not_edges = 0.8;
	double threshold_ratio = 0.4;

	int total = 0;
	high = 0;
	// size.height should be here too, but right now we're going row-by-row
	while (total < size.height * size.width * percent_of_pixels_not_edges) {
		total+= bins[high];
		high++;
	}

	high *= bin_size;
	low = threshold_ratio * high;
	cout << "high: " << high << endl;
	cout << "low: " << low << endl;
	int adit = 10;

	// non-maxima suppression
	
    for( i = 1; i <= size.height; i++ )
    {
		int *_mag = mag + (size.width + 2) * i;
        if( (stack_top - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }
		const short* _dx = (short*)(dx->data.ptr + dx->step*(i-1));
        const short* _dy = (short*)(dy->data.ptr + dy->step*(i-1));
	    int prev_flag = 0;
		int x, y;

        uchar* _map;
        ptrdiff_t magstep1, magstep2;

        _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;

		if (i % 3 == 1) {
			magstep1 = size.width + 2;
			magstep2 = -(size.width + 2);
		} else if (i % 3 == 2) {
			magstep1 = -2 * (size.width + 2);
			magstep2 = -(size.width + 2);
		} else {
			magstep1 =  size.width + 2;
			magstep2 = 2 * (size.width + 2);
		}


        for( j = 0; j < size.width; j++ )
        {
            #define CANNY_SHIFT 15

			// i.e. tan(pi/8) * (1 << CANNY_SHIFT etc...)
            #define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

            x = _dx[j];
            y = _dy[j];

            int s = x ^ y;
            int m = _mag[j];

            x = abs(x);
            y = abs(y);

            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else {
                            _map[j] = (uchar)0;
						}
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j-magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else {
                            _map[j] = (uchar)0;
						}							
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else {
                            _map[j] = (uchar)0;
						}
                        continue;
                    }
                }
			}
            prev_flag = 0;
            _map[j] = (uchar)1;
		}
	}	
    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( (stack_top - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);
    
        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep-1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep+1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep-1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep+1] )
            CANNY_PUSH( m + mapstep + 1 );
    }

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep*(i+1) + 1;
        uchar* _dst = dst->data.ptr + dst->step*i;
        
        for( j = 0; j < size.width; j++ )
            _dst[j] = (uchar)-(_map[j] >> 1);
    }
	adit = 20;
}

void cv::Canny( const Mat& image, Mat& edges,
                double threshold1, double threshold2,
                int apertureSize, bool L2gradient )
{
    Mat src = image;
    edges.create(src.size(), CV_8U);
    CvMat _src = src, _dst = edges;
    cvCanny( &_src, &_dst, threshold1, threshold2,
        apertureSize + (L2gradient ? CV_CANNY_L2_GRADIENT : 0));
}

/* End of file. */





