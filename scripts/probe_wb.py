import eBUS as eb
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("probe_wb")

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
        log.error("Camera not found!")
        return
        
    log.info("Connecting to camera...")
    result, device = eb.PvDevice.CreateAndConnect(connection_id)
    if not result.IsOK() or device is None:
        log.error("Failed to connect!")
        return
        
    try:
        nm = device.GetParameters()
        log.info("\n--- GENICAM PARAMETERS CONTAINING 'BALANCE', 'RATIO', 'WHITE', or 'GAIN' ---")
        
        # We temporarily select Source0
        stack = eb.PvGenStateStack(nm)
        stack.SetEnumValue("SourceSelector", "Source0")
        
        for k in range(nm.GetCount()):
            p = nm.Get(k)
            name = p.GetName()[1]
            if any(x in name.lower() for x in ["balance", "ratio", "white", "gain"]):
                # Get the current value if possible
                val_str = "N/A"
                try:
                    _, val_str = p.GetValueString()
                except Exception:
                    pass
                log.info("Node: %s (%s) = %s", name, p.GetInterfaceType(), val_str)
                
                # If it's an enum, print its entry options
                if p.GetInterfaceType() == 2:  # PvGenTypeEnum
                    try:
                        enum_node = nm.GetEnum(name)
                        count = enum_node.GetEntriesCount()[1]
                        entries = []
                        for idx in range(count):
                            entries.append(enum_node.GetEntryByIndex(idx)[1].GetName()[1])
                        log.info("   -> Enum options: %s", entries)
                    except Exception as e:
                        log.info("   -> Could not list enum options: %s", e)
                        
    finally:
        device.Disconnect()
        eb.PvDevice.Free(device)
        log.info("Disconnected cleanly.")

if __name__ == "__main__":
    main()
