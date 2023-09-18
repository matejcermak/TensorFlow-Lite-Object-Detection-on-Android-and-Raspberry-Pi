import logging
import time

from openant.easy.node import Node
from openant.easy.channel import Channel
from openant.base.commons import format_list
from functions import *

# Definition of Variables
NETWORK_KEY = [0xB9, 0xA5, 0x21, 0xFB, 0xBD, 0x72, 0xC3, 0x45]
Device_Type = 40  # 40 = bike radar
device_number = 123  # Change if you need.
Channel_Period = 4084
Channel_Frequency = 57

##########################################################################


class RadarBroadcast:
    def __init__(self, dst, dng):
        self.message_count = 0
        self.message_payload = [0, 0, 0, 0, 0, 0, 0, 0]
        self.time_program_start = time.time()
        self.dst = dst
        self.dng = dng

    def yield_datapage(self):
        # Define Variables
        self.message_count += 1
        # encode 0-1 distance to 6bit distance
        # 45 turned out to be longest visible distance (not 63 - 0b111111)
        distance = round(self.dst.value * 45) if self.dst.value != 1.0 else 63
        # distance = 45
        danger = self.dng.value

        if self.message_count == 1:
            self.message_payload[0] = 80  # common page 80 - manufacturer info
            self.message_payload[1] = 0xFF
            self.message_payload[2] = 0xFF  # Reserved
            self.message_payload[3] = 1  # SW Revision
            self.message_payload[4] = 0xFF  # Manufacturer ID LSB
            self.message_payload[5] = 0xFF  # Manufacturer ID MSB
            self.message_payload[6] = 0xFF  # Model Number LSB
            self.message_payload[7] = 0xFF  # Model Number MSB
        elif self.message_count == 63:
            self.message_payload[0] = 81  # common page 82 - product information
            self.message_payload[1] = 0xFF
            self.message_payload[2] = 0xFF  # SW Revision (supplemental)
            self.message_payload[3] = 1  # SW Revision (Main)
            self.message_payload[4] = 0xFF  # Serial Number
            self.message_payload[5] = 0xFF
            self.message_payload[6] = 0xFF
            self.message_payload[7] = 0xFF
        elif self.message_count == 125:
            self.message_payload[0] = 82  # common page 82 - battery info
            self.message_payload[1] = 0xFF
            self.message_payload[2] = 0xFF  # battery ID
            self.message_payload[3] = 0xFF  # cummulative operating time
            self.message_payload[4] = 0xFF
            self.message_payload[5] = 0xFF
            self.message_payload[6] = 0x8B  # fractional battery voltage
            self.message_payload[7] = 0x32  # descriptive bit - custom value from example
        # elif distance >= 63:  # simulate no cars - clear display
        #     self.message_payload[0] = 1  # device status
        #     self.message_payload[1] = 0b11111100  # 2 bits - device state, 6bits - reserved
        #     self.message_payload[2] = 0xFF  # battery ID
        #     self.message_payload[3] = 0xFF  # reserved
        #     self.message_payload[4] = 0xFF
        #     self.message_payload[5] = 0xFF
        #     self.message_payload[6] = 0xFF
        #     self.message_payload[7] = 0b11111110  # last 1bit inverted clear targets
        else:
            # [0] - Data page number
            self.message_payload[0] = 48  # Radar Targets 1-4
            # [1] - (2bits/target) Threat Level Target 1-4
            # 0 - no threat, 1 - vehicle approach, 2 - vehicle fast approach, 3 reserved
            if danger > DANGER_THRESHOLD:  # danger triggered > trigger "fast approach"
                #                           aabbccdd
                self.message_payload[1] = 0b10000000
            else:
                self.message_payload[1] = 0b01000000
            # [2] (2bits/target) Threat Side Target 1-4
            # 0 - behind (noside), 1 - right, 2 - left, 3 - reserved
            self.message_payload[2] = 0b00000000
            # [3-5] (6bits/target) Range Target
            #                             ccdddddd
            # self.message_payload[3] = 0b00100000
            #                             bbbbcccc
            # self.message_payload[4] = 0b10000100
            #                             aaaaaabb
            # self.message_payload[5] = 0b00010000
            self.message_payload[3] = 0x00
            self.message_payload[4] = 0x00
            self.message_payload[5] = distance << 2  # simulate moving car #1
            # [6-7] (4bits/target) Closing speed target - needed?
            self.message_payload[6] = 0b11110000
            self.message_payload[7] = 0b00000000

        # rollover counter
        if self.message_count >= 190:
            self.message_count = 0

        return self.message_payload

    # TX Event
    def on_event_tx(self, data):
        message_payload = self.yield_datapage()
        self.actual_time = time.time() - self.time_program_start

        self.channel.send_broadcast_data(self.message_payload)  # Final call for broadcasting data

        # print debug message
        print(
            round(self.actual_time, 1),
            "TX:",
            device_number,
            ",",
            message_payload[0],
            ":",
            format_list(message_payload),
            " - ",
            f"dst: {self.dst.value}, dng: {self.dng.value}",
        )

    # Open Channel
    def open_channel(self):
        self.node = Node()  # initialize the ANT+ device as node

        # CHANNEL CONFIGURATION
        self.node.set_network_key(0x00, NETWORK_KEY)  # set network key
        self.channel = self.node.new_channel(
            Channel.Type.BIDIRECTIONAL_TRANSMIT, 0x00, 0x00
        )  # Set Channel, Master TX
        self.channel.set_id(
            device_number, Device_Type, 5
        )  # set channel id as <Device Number, Device Type, Transmission Type>
        self.channel.set_period(Channel_Period)  # set Channel Period
        self.channel.set_rf_freq(Channel_Frequency)  # set Channel Frequency

        # Callback function for each TX event
        self.channel.on_broadcast_tx_data = self.on_event_tx

        try:
            self.channel.open()  # Open the ANT-Channel with given configuration
            self.node.start()
        except KeyboardInterrupt:
            print("Closing ANT+ Channel...")
            self.channel.close()
            self.node.stop()
        finally:
            print("Final checking...")
            # not sure if there is anything else we should check?! :)
