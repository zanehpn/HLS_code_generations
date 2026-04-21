#include <stdint.h>

#define PACKET_SIZE 1024

void ethernet_mac_controller(uint8_t input_packet[PACKET_SIZE], uint8_t output_packet[PACKET_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output_packet type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input_packet type=cyclic dim=1 factor=16
    uint16_t crc = 0xFFFF;
    uint8_t polynomial = 0x07;

    // Calculate CRC for input packet
    for (int i = 0; i < PACKET_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        crc ^= input_packet[i];
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
            if (crc & 1) {
                crc = (crc >> 1) ^ polynomial;
            } else {
                crc >>= 1;
            }
        }
    }

    // Append CRC to the output packet
    for (int i = 0; i < PACKET_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        output_packet[i] = input_packet[i];
    }
    output_packet[PACKET_SIZE - 2] = crc & 0xFF;
    output_packet[PACKET_SIZE - 1] = (crc >> 8) & 0xFF;
}

// Top function name: ethernet_mac_controller
