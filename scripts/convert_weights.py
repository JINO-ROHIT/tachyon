'''this script converts model weights from safetensor format to .tach format'''

import argparse
import os
import json
import mmap
import numpy as np

from huggingface_hub import snapshot_download

class WeightMapper:
    '''
    this class downloads the model from huggingface and convert the weight format from safetensors to .tach format.
    1. model.tach (fp32/fp16 tensors, 32 bit aligned)
    2. model_index.json (tensor metadata for loading)
    '''
    def __init__(self, model_name: str, output_name: str = "model.tach"):
        self.model_name = model_name
        self.model_path = None
        self.config = None
        self.safetensors_files = []

        self.m = None
        self.metadata = {}

        self.header_offset = 0
        self.bin_offset = 0

        self.layer_index = {}

        self._download_model()
        self.output_name = f"models/{self.model_name}/{output_name}"

        os.makedirs(f"models/{self.model_name}", exist_ok=True)
        self.bin_file = open(self.output_name, "wb")
    
    def _get_hex(self, x):
        """store offsets as hex addresses"""
        return f"0x{x:08X}"

    def _download_model(self):
        """download the model from hf"""
        self.model_path = snapshot_download(self.model_name)
        self.config = f"{self.model_path}/config.json"
        self.safetensors_files = self._find_safetensors_files()

    def _find_safetensors_files(self):
        """find all .safetensors files in the model directory"""
        safetensors_files = []
        for file in os.listdir(self.model_path):
            if file.endswith('.safetensors'):
                safetensors_files.append(os.path.join(self.model_path, file))
        return sorted(safetensors_files)
    
    def _calculate_offsets(self, layer_name):
        """collect layer wise information"""
        offs = self.metadata[layer_name]['data_offsets']
        shape = self.metadata[layer_name]['shape']
        dtype = self.metadata[layer_name]['dtype']

        off = self.header_offset + offs[0] # basically the header + starting position
        size = offs[1] - offs[0] # ending - starting
        buffer = self.m[off:off+size]
        
        padded_size = self._store_weights(layer_name, buffer)
        
        # Store layer metadata
        self.layer_index[layer_name] = {
            'offset': self._get_hex(self.bin_offset),
            'size': size,
            'padded_size': padded_size,
            'shape': shape,
            'dtype': dtype,
            'transposed': False
        }
        
        return padded_size
    
    # to-do figure out if any layers are transposed
    def _calculate_offsets_transposed(self, layer_name):
        offs = self.metadata[layer_name]['data_offsets']
        shape = self.metadata[layer_name]['shape']
        dtype = self.metadata[layer_name]['dtype']

        off = self.header_offset + offs[0] # basically the header + starting position
        size = offs[1] - offs[0] # ending - starting

        w = np.frombuffer(self.m[off: off + size], dtype=np.float32)
        w = w.reshape(self.metadata[layer_name]["shape"])
        w = w.T.copy()
        
        transposed_buffer = w.tobytes()
        padded_size = self._store_weights(layer_name, transposed_buffer)
        
        self.layer_index[layer_name] = {
            'offset': self._get_hex(self.bin_offset),
            'size': len(transposed_buffer),
            'padded_size': padded_size,
            'shape': list(reversed(shape)), 
            'dtype': dtype,
            'transposed': True
        }
        
        return padded_size
    
    def _store_weights(self, layer_name, buff):
        self.bin_file.write(buff)

        buff_size = len(buff)
        padded_size = (buff_size + 31) & (~31) # pad weights to align with 32 bytes
        if padded_size > buff_size:
            print(f"adding padding to {layer_name}")
            self.bin_file.write(bytearray(padded_size - buff_size)) # write the null bytes as padding at the end
        
        return padded_size

    def do_mmap(self):
        """process multiple safetensors files and convert weights"""
        print(f"Found {len(self.safetensors_files)} safetensors files:")
        for st_file in self.safetensors_files:
            print(f"  - {os.path.basename(st_file)}")

        for safetensors_file in self.safetensors_files:
            print(f"\nProcessing {os.path.basename(safetensors_file)}...")

            with open(safetensors_file, 'r') as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as self.m:
                    # check here for format - https://github.com/huggingface/safetensors
                    header = self.m.read(8)
                    n = int.from_bytes(header, byteorder="little")
                    metadata_bytes = self.m.read(n) # advance to metadata
                    file_metadata = json.loads(metadata_bytes)
                    self.header_offset = n + 8 # distance away from header

                    # process each layer in this safetensors file
                    for layer_name, layer_data in file_metadata.items():
                        if layer_name == '__metadata__':
                            continue
                        else:
                            self.metadata[layer_name] = layer_data
                            self.bin_offset += self._calculate_offsets(layer_name)

        self.bin_file.close()
        self._save_index()
        self._add_config_info()
        return self.bin_offset
    
    def _save_index(self):
        index_filename = self.output_name.replace('.tach', '_index.json')
        with open(index_filename, 'w') as f:
            json.dump(self.layer_index, f, indent=2)
        print(f"Saved layer index to {index_filename}")
    
    def _add_config_info(self):
        index_filename = self.output_name.replace('.tach', '_index.json')

        with open(self.config, 'r') as f:
            config_data = json.load(f)

        with open(index_filename, 'r') as f:
            data = json.load(f)

        # convert each config value to its own dictionary for better JSON/nlohmann compatibility
        config_dict = {}
        for key, value in config_data.items():
            config_dict[key] = {"value": value}

        data["config"] = config_dict

        with open(index_filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Added config information to {index_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="tachyon weight converter",
        description="a utility to convert model weights to .tach format"
    )
    parser.add_argument("model")
    args = parser.parse_args()

    mapper = WeightMapper(args.model) # Qwen/Qwen3-0.6B
    total_size = mapper.do_mmap()
    print(f"Total binary size: {total_size} bytes")
    print(f"Layers tracked: {len(mapper.layer_index)}")