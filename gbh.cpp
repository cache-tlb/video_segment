/*
Original Code From:
Copyright (C) 2006 Pedro Felzenszwalb
Modifications (may have been made) Copyright (C) 2011, 2012
  Chenliang Xu, Jason Corso.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "dirent.h"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "image.h"
#include "pnmfile.h"
#include "segment-image.h"
#include "disjoint-set.h"

#include "api.h"

image<rgb>* cvmat_to_image(const cv::Mat &mat) {
    int width = mat.cols, height = mat.rows;
    image<rgb> *im = new image<rgb>(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b color = mat.at<cv::Vec3b>(y, x);
            rgb im_color;
            im_color.r = color[2];
            im_color.g = color[1];
            im_color.b = color[0];
            imRef(im, x, y) = im_color;
        }
    }
    return im;
}

bool GraphBasedHierarchicalSegmentation(const std::vector<cv::Mat> &frames, float c, float c_reg, int min_size, float sigma, int hie_num, int output_level, std::vector<cv::Mat> &output_frame_labels) {
    if (c <= 0 || c_reg < 0 || min_size < 0 || sigma < 0 || hie_num < 0) {
        fprintf(stderr, "Unable to use the input parameters.");
        return false;
    }
    int num_frame = frames.size();
    if (num_frame == 0) {
        fprintf(stderr, "empty video.");
        return false;
    }
    printf("Total number of frames in fold is %d\n", num_frame);
    image<rgb>** im = new image<rgb>*[num_frame];
    for (int i = 0; i < num_frame; i++) {
        im[i] = cvmat_to_image(frames[i]);
    }

    // segmentation
    // segment_image(output_path, images, frame_num, c, c_reg, min_size, sigma, hie_num);

    // step 1 -- Get information
    int width = im[0]->width();
    int height = im[0]->height();
    // ----- node number
    int num_vertices = num_frame * width * height;
    // ----- edge number
    int num_edges_plane = (width - 1) * (height - 1) * 2 + width * (height - 1)
        + (width - 1) * height;
    int num_edges_layer = (width - 2) * (height - 2) * 9 + (width - 2) * 2 * 6
        + (height - 2) * 2 * 6 + 4 * 4;
    int num_edges = num_edges_plane * num_frame
        + num_edges_layer * (num_frame - 1);
    // ----- hierarchy setup
    std::vector<std::vector<edge>*> edges_region;
    edges_region.resize(hie_num + 1);
    // ------------------------------------------------------------------

    // step 2 -- smooth images
    image<float>** smooth_r = new image<float>*[num_frame];
    image<float>** smooth_g = new image<float>*[num_frame];
    image<float>** smooth_b = new image<float>*[num_frame];
    smooth_images(im, num_frame, smooth_r, smooth_g, smooth_b, sigma);
    // ------------------------------------------------------------------

    // step 3 -- build edges
    printf("start build edges\n");
    edge* edges = new edge[num_edges];
    initialize_edges(edges, num_frame, width, height, smooth_r, smooth_g,
        smooth_b);
    printf("end build edges\n");
    // ------------------------------------------------------------------

    // step 4 -- build nodes
    printf("start build nodes\n");
    universe* mess = new universe(num_frame, width, height, smooth_r, smooth_g,
        smooth_b, hie_num);
    printf("end build nodes\n");
    // ------------------------------------------------------------------

    // step 5 -- over-segmentation
    printf("start over-segmentation\n");
    edges_region[0] = new vector<edge>();
    segment_graph(mess, edges_region[0], edges, num_edges, c, 0);
    // optional merging small components
    for (int i = 0; i < num_edges; i++) {
        int a = mess->find_in_level(edges[i].a, 0);
        int b = mess->find_in_level(edges[i].b, 0);
        if ((a != b)
            && ((mess->get_size(a) < min_size)
            || (mess->get_size(b) < min_size)))
            mess->join(a, b, 0, 0);
    }
    printf("end over-segmentation\n");
    // ------------------------------------------------------------------

    // step 6 -- hierarchical segmentation
    for (int i = 0; i < hie_num; i++) {
        printf("level = %d\n", i);
        // incremental in each hierarchy
        min_size = min_size * 1.2;
        printf("start update\n");
        mess->update(i);
        printf("end update\n");

        printf("start fill edge weight\n");
        fill_edge_weight(*edges_region[i], mess, i);
        printf("end fill edge weight\n");

        printf("start segment graph region\n");
        edges_region[i + 1] = new vector<edge>();
        segment_graph_region(mess, edges_region[i + 1], edges_region[i], c_reg, i + 1);
        printf("end segment graph region\n");

        printf("start merging min_size\n");
        for (int it = 0; it < (int) edges_region[i]->size(); it++) {
            int a = mess->find_in_level((*edges_region[i])[it].a, i + 1);
            int b = mess->find_in_level((*edges_region[i])[it].b, i + 1);
            if ((a != b)
                && ((mess->get_size(a) < min_size)
                || (mess->get_size(b) < min_size)))
                mess->join(a, b, 0, i + 1);
        }
        printf("end merging min_size\n");

        c_reg = c_reg * 1.4;
        delete edges_region[i];
    }
    delete edges_region[hie_num];
    // ------------------------------------------------------------------

    // step 8 -- generate output
    // generate_output(path, num_frame, width, height, mess, num_vertices, hie_num);
    // here only one level of hierarchy is output
    int k = output_level;
    output_frame_labels.clear();
    output_frame_labels.resize(num_frame);
    for (int i = 0; i < num_frame; i++) {
        cv::Mat frame_label(height, width, CV_32S);
        int min_comp = 1 << 30, max_comp = 0;
        std::set<int> comps;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int comp = mess->find_in_level(
                    y * width + x + i * (width * height), k);
                frame_label.at<int>(y, x) = comp;
                min_comp = std::min(min_comp, comp);
                max_comp = std::max(max_comp, comp);
                comps.insert(comp);
            }
        }
        printf("min_comp: %d, max_comp: %d, comps: %d\n", min_comp, max_comp, comps.size());
        output_frame_labels[i] = frame_label;
    }
    // ------------------------------------------------------------------

    // step 9 -- clear everything
    delete mess;
    delete[] edges;
    for (int i = 0; i < num_frame; i++) {
        delete smooth_r[i];
        delete smooth_g[i];
        delete smooth_b[i];
    }
    delete[] smooth_r;
    delete[] smooth_g;
    delete[] smooth_b;
    for (int i = 0; i < num_frame; i++) {
        delete im[i];
    }
    delete[] im;
}

void test_api() {
    // std::string video_path = "E:\\liubin\\temp\\videos\\One_Evening.2.2.mp4";
    std::string video_path = "E:/liubin/temp/videos/Vid_K_cup_on_table.avi";
    int begin = 0, end  = 400;

    cv::VideoCapture vc(video_path);
    if(!vc.isOpened()) {
        std::cout<<"fail to open!"<< std::endl;
    }
    long totalFrameNumber = vc.get(CV_CAP_PROP_FRAME_COUNT);
    std::cout << "frame count: " << totalFrameNumber << std::endl;
    std::vector<cv::Mat> frames;
    frames.clear();
    cv::Mat tmp;
    for (int i = begin; i < end; i++) {
        // std::cout<< i <<std::endl;
        vc.set(CV_CAP_PROP_POS_FRAMES, i);
        vc.read(tmp);
        if (tmp.rows == 0 || tmp.cols == 0) continue;
        cv::resize(tmp, tmp, cv::Size(tmp.cols / 2, tmp.rows / 2));
        frames.push_back(tmp.clone());
    }
    std::vector<cv::Mat> frame_labels;
    float c = 2, c_reg = 200, sigma = 0.5;
    int min_size = 100;
    GraphBasedHierarchicalSegmentation(frames, c, c_reg, min_size, sigma, 23, 23, frame_labels);

    // show segments
    vector<int> randomNumbers;
    juRandomPermuteRange(16777215, randomNumbers, NULL);
    for (int i = 0; i < frame_labels.size(); i++) {
        cv::Mat show_im(frame_labels[i].size(), CV_8UC3);
        for (int k = 0; k < frame_labels[i].rows * frame_labels[i].cols; k++) {
            int label = frame_labels[i].at<int>(k);
            show_im.at<cv::Vec3b>(k) = cv::Vec3b((label >> 16) % 256, (label >> 8) % 256, label % 256);
        }
        cv::imshow("seg", show_im);
        cv::waitKey();
    }
}

int main(int argc, char **argv) {
    test_api();
    return 0;
	if (argc != 8) {
		printf("%s c c_reg min sigma hie_num input output\n", argv[0]);
		printf("       c --> value for the threshold function in over-segmentation\n");
		printf("   c_reg --> value for the threshold function in hierarchical region segmentation\n");
		printf("     min --> enforced minimum supervoxel size\n");
		printf("   sigma --> variance of the Gaussian smoothing.\n");
		printf(" hie_num --> desired number of hierarchy levels\n");
		printf("   input --> input path of ppm video frames\n");
		printf("  output --> output path of segmentation results\n");
		return 1;
	}

	// Read Parameters
	float c = atof(argv[1]);
	float c_reg = atof(argv[2]);
	int min_size = atoi(argv[3]);
	float sigma = atof(argv[4]);
	int hie_num = atoi(argv[5]);
	char* input_path = argv[6];
	char* output_path = argv[7];
	if (c <= 0 || c_reg < 0 || min_size < 0 || sigma < 0 || hie_num < 0) {
		fprintf(stderr, "Unable to use the input parameters.");
		return 1;
	}

	// count files in the input directory
	int frame_num = 0;
	struct dirent* pDirent;
	DIR* pDir;
	pDir = opendir(input_path);
	if (pDir != NULL) {
		while ((pDirent = readdir(pDir)) != NULL) {
			int len = strlen(pDirent->d_name);
			if (len >= 4) {
				if (strcmp(".ppm", &(pDirent->d_name[len - 4])) == 0)
					frame_num++;
			}
		}
	}
	if (frame_num == 0) {
		fprintf(stderr, "Unable to find video frames at %s", input_path);
		return 1;
	}
	printf("Total number of frames in fold is %d\n", frame_num);


	// make the output directory
	struct stat st;
	int status = 0;
	char savepath[1024];
  	snprintf(savepath,1023,"%s",output_path);
	if (stat(savepath, &st) != 0) {
		/* Directory does not exist */
		if (mkdir(savepath, S_IRWXU) != 0) {
			status = -1;
		}
	}
	for (int i = 0; i <= hie_num; i++) {
  		snprintf(savepath,1023,"%s/%02d",output_path,i);
		if (stat(savepath, &st) != 0) {
			/* Directory does not exist */
			if (mkdir(savepath, S_IRWXU) != 0) {
				status = -1;
			}
		}
	}
	if (status == -1) {
		fprintf(stderr,"Unable to create the output directories at %s",output_path);
		return 1;
	}


	// Initialize Parameters
	image<rgb>** images = new image<rgb>*[frame_num];
	char filepath[1024];

	// Time Recorder
	time_t Start_t, End_t;
	int time_task;
	Start_t = time(NULL);

	// Read Frames
	for (int i = 0; i < frame_num; i++) {
		snprintf(filepath, 1023, "%s/%05d.ppm", input_path, i + 1);
		images[i] = loadPPM(filepath);
		printf("load --> %s\n", filepath);
	}

	// segmentation
	segment_image(output_path, images, frame_num, c, c_reg, min_size, sigma, hie_num);

	// Time Recorder
	End_t = time(NULL);
	time_task = difftime(End_t, Start_t);
	std::ofstream myfile;
	char timefile[1024];
	snprintf(timefile, 1023, "%s/%s", output_path, "time.txt");
	myfile.open(timefile);
	myfile << time_task << endl;
	myfile.close();

	printf("Congratulations! It's done!\n");
	printf("Time_total = %d seconds\n", time_task);
	return 0;
}


