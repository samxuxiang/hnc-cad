import os
import argparse
import ntpath
from tqdm import tqdm
import multiprocessing
from pathlib import Path
from glob import glob
import trimesh
from trimesh.sample import sample_surface
from plyfile import PlyData, PlyElement
import numpy as np


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


def find_files(folder, extension):
    return sorted([Path(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith(extension)])

class SamplePoints:
    """
    Perform sampleing of points.
    """

    def __init__(self):
        """
        Constructor.
        """
        parser = self.get_parser()
        self.options = parser.parse_args()


    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """
        parser = argparse.ArgumentParser(description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--in_dir', type=str, help='Path to input directory.')
        parser.add_argument('--out_dir', type=str, help='Path to output directory; files within are overwritten!')
        return parser


    def run_parallel(self, project_folder):
        out_folder =  os.path.join(project_folder, self.options.out_dir)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        files = find_files(project_folder, 'final.stl')
        
        for filepath in files:
            N_POINTS = 2000
            try:
                out_mesh = trimesh.load(str(filepath))
                out_pc, _ = sample_surface(out_mesh, N_POINTS)
                save_path = os.path.join(out_folder, ntpath.basename(filepath)[:-4]+'_pcd.ply')
                write_ply(out_pc, save_path)

            except Exception as ex:        
                return project_folder
        return


    def run(self):
        """
        Run simplification.
        """
        project_folders = sorted(glob(self.options.in_dir+'/*/'))
        num_cpus = multiprocessing.cpu_count()
        convert_iter =  multiprocessing.Pool(num_cpus).imap(self.run_parallel, project_folders) 
        for _ in tqdm(convert_iter, total=len(project_folders)):
            pass
       

if __name__ == '__main__':
    app = SamplePoints()
    app.run()
