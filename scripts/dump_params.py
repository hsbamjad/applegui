import eBUS as eb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dump_params")

def main():
    log.info("Scanning for FS-3200T camera...")
    sys_obj = eb.PvSystem()
    sys_obj.Find()
    
    connection_id = None
    target_mac = "00:0c:df:0a:b8:e9"
    
    for i in range(sys_obj.GetInterfaceCount()):
        iface = sys_obj.GetInterface(i)
        for j in range(iface.GetDeviceCount()):
            dev = iface.GetDeviceInfo(j)
            mac = str(dev.GetMACAddress())
            if target_mac.lower() in mac.lower():
                connection_id = dev.GetConnectionID()
                break
        if connection_id:
            break
            
    if connection_id is None:
        log.error("Camera not found on network!")
        return
        
    log.info("Connecting to camera...")
    result, device = eb.PvDevice.CreateAndConnect(connection_id)
    if not result.IsOK() or device is None:
        log.error("Failed to connect!")
        return
        
    out_file = Path("scripts/camera_params.txt")
    log.info("Dumping all GenICam parameters to %s ...", out_file.resolve())
    
    try:
        nm = device.GetParameters()
        
        # Temporarily select Source0 (color channel) to expose its sub-channel registers
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", "Source0")
        
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"FS-3200T GENICAM PARAMETERS DUMP\n")
            f.write(f"="*80 + "\n\n")
            
            for k in range(nm.GetCount()):
                p = nm.Get(k)
                name = p.GetName()[1]
                p_type = p.GetInterfaceType()
                
                # Try to get value as string
                val_str = "N/A"
                try:
                    _, val_str = p.GetValueString()
                except Exception:
                    pass
                
                f.write(f"Node: {name}\n")
                f.write(f"  Type: {p_type}\n")
                f.write(f"  Value: {val_str}\n")
                
                # Detailed Enum options if applicable
                if p_type == 2:  # PvGenTypeEnum
                    try:
                        enum_node = nm.GetEnum(name)
                        count = enum_node.GetEntriesCount()[1]
                        entries = []
                        for idx in range(count):
                            entries.append(enum_node.GetEntryByIndex(idx)[1].GetName()[1])
                        f.write(f"  Enum Options: {entries}\n")
                    except Exception:
                        pass
                f.write("\n")
                
        log.info("Successfully dumped %d parameters to %s!", nm.GetCount(), out_file)
    except Exception as e:
        log.error("Exception during parameter dump: %s", e)
    finally:
        device.Disconnect()
        eb.PvDevice.Free(device)
        log.info("Disconnected cleanly.")

if __name__ == "__main__":
    main()
